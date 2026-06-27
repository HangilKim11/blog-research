"""Linear probe: frozen SSL encoder + logistic regression on customer features.

Loads a pretrained encoder checkpoint, extracts mean-pooled embeddings for
every customer (train + val + test), fits a logistic regression on
trainval, then scores the held-out test set. Mirrors the Phase 1 baseline
protocol so the numbers are directly comparable.

Output JSON shape mirrors ``oof_lgbm_metrics.json`` to make the comparison
script trivial.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import typer
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

from amex.evaluation.metrics import amex_metric_components
from amex.models.transformer import EncoderConfig, SequenceEncoder
from amex.ssl.dataset import AmexSSLDataset, collate_ssl_samples
from amex.ssl.tokenizer import TokenizerArtifact

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=False)

KEY_COL = "customer_ID"
TARGET_COL = "target"
SPLIT_COL = "split"
FOLD_COL = "fold"
TEST_SPLIT = "test"


def _load_encoder(ckpt_path: Path, tokenizer: TokenizerArtifact) -> SequenceEncoder:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = payload["encoder_cfg"]
    enc_cfg = EncoderConfig(**cfg_dict)
    enc = SequenceEncoder(tokenizer, enc_cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    enc.eval()
    return enc


def _embed_customers(
    encoder: SequenceEncoder,
    tokenizer: TokenizerArtifact,
    train_glob: str,
    customer_ids: set[str],
    *,
    batch_size: int = 512,
    device: str = "cuda",
) -> tuple[list[str], np.ndarray]:
    """Run the encoder on every requested customer, return (ids, mean-pooled features)."""
    ds = AmexSSLDataset(
        partition_glob=train_glob,
        tokenizer=tokenizer,
        customer_id_filter=customer_ids,
        shuffle_partitions=False,
        shuffle_within_partition=False,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_ssl_samples,
    )

    encoder = encoder.to(device)
    encoder.eval()

    ids: list[str] = []
    feats: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            num_vals = batch["numeric_values"].to(device, non_blocking=True)
            num_mask = batch["numeric_mask"].to(device, non_blocking=True)
            cat_ids = batch["categorical_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(
                device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16
            ):
                hidden = encoder(num_vals, num_mask, cat_ids, attn)  # (B, T', D)
            # mean-pool over the per-timestep slots (drop CLS at index 0 if present)
            per_t = hidden[:, 1:, :] if encoder.cfg.use_cls_token else hidden
            # mean-pool with attention_mask (T positions; CLS isn't in mask either way here)
            mask = attn.unsqueeze(-1).to(per_t.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (per_t * mask).sum(dim=1) / denom  # (B, D)
            feats.append(pooled.float().cpu().numpy())
            ids.extend(batch["customer_id"])

    return ids, np.concatenate(feats, axis=0)


def _ks(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = np.sort(y_pred[y_true == 1])
    neg = np.sort(y_pred[y_true == 0])
    if pos.size == 0 or neg.size == 0:
        return 0.0
    grid = np.concatenate([pos, neg])
    cdf_pos = np.searchsorted(pos, grid, side="right") / pos.size
    cdf_neg = np.searchsorted(neg, grid, side="right") / neg.size
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def _all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    m, g, d = amex_metric_components(y_true, y_pred)
    return {
        "amex": m,
        "amex_g": g,
        "amex_d": d,
        "auc": float(roc_auc_score(y_true, y_pred)),
        "ks": _ks(y_true, y_pred),
        "log_loss": float(log_loss(y_true, np.clip(y_pred, 1e-15, 1 - 1e-15))),
    }


def _five_fold_oof(
    X: np.ndarray, y: np.ndarray, folds: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Use the existing fold column (0..4) for OOF; matches the Phase 1 protocol."""
    oof = np.zeros(len(y), dtype=np.float64)
    for k in sorted(set(folds.tolist()) - {-1}):
        tr = folds != k
        va = folds == k
        clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs", n_jobs=-1)
        clf.fit(X[tr], y[tr])
        oof[va] = clf.predict_proba(X[va])[:, 1]
    return oof, _all_metrics(y, oof)


@app.command()
def main(
    encoder_ckpt: Path = typer.Option(
        ..., "--encoder", help="encoder.pt produced by ssl.pretrain."
    ),
    tokenizer_path: Path = typer.Option(Path("data/processed/v1/tokenizer.json"), "--tokenizer"),
    splits_path: Path = typer.Option(Path("data/splits/v1.parquet"), "--splits"),
    train_glob: str = typer.Option("data/processed/v1/train/**/*.parquet", "--train-glob"),
    out_dir: Path = typer.Option(Path("data/processed/v1/probes"), "--out"),
    batch_size: int = typer.Option(512, "--batch-size"),
    device: str = typer.Option("cuda", "--device"),
) -> None:
    """Linear probe: encode all customers, fit LR, dump metrics + OOF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = TokenizerArtifact.load(tokenizer_path)
    encoder = _load_encoder(encoder_ckpt, tokenizer)
    n_enc_params = sum(p.numel() for p in encoder.parameters())
    console.print(f"[bold]loaded encoder from {encoder_ckpt} -- {n_enc_params:,} params[/]")

    splits = pl.read_parquet(splits_path).select([KEY_COL, TARGET_COL, SPLIT_COL, FOLD_COL])
    trainval_ids = set(splits.filter(pl.col(SPLIT_COL) != TEST_SPLIT)[KEY_COL].to_list())
    test_ids = set(splits.filter(pl.col(SPLIT_COL) == TEST_SPLIT)[KEY_COL].to_list())

    # --- embed ---
    t0 = time.monotonic()
    trainval_id_list, X_tv = _embed_customers(
        encoder, tokenizer, train_glob, trainval_ids, batch_size=batch_size, device=device
    )
    t_train_embed = time.monotonic() - t0
    console.print(f"  trainval embedded: {X_tv.shape} in {t_train_embed:.1f}s")

    t0 = time.monotonic()
    test_id_list, X_te = _embed_customers(
        encoder, tokenizer, train_glob, test_ids, batch_size=batch_size, device=device
    )
    t_test_embed = time.monotonic() - t0
    console.print(f"  test embedded:     {X_te.shape} in {t_test_embed:.1f}s")

    # join targets and folds back
    splits_df = splits.to_pandas().set_index(KEY_COL)
    y_tv = splits_df.loc[trainval_id_list][TARGET_COL].to_numpy()
    fold_tv = splits_df.loc[trainval_id_list][FOLD_COL].to_numpy()
    y_te = splits_df.loc[test_id_list][TARGET_COL].to_numpy()

    # --- 5-fold OOF on trainval ---
    oof_pred, oof_metrics = _five_fold_oof(X_tv, y_tv, fold_tv)

    # --- fit on full trainval, score test ---
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", n_jobs=-1)
    clf.fit(X_tv, y_tv)
    test_pred = clf.predict_proba(X_te)[:, 1]
    test_metrics = _all_metrics(y_te, test_pred)

    summary: dict[str, Any] = {
        "encoder_ckpt": str(encoder_ckpt),
        "n_encoder_params": n_enc_params,
        "embed_dim": int(X_tv.shape[1]),
        "oof": {**oof_metrics, "n": len(y_tv)},
        "test": {**test_metrics, "n": len(y_te)},
        "embedding_wall_time_s": float(t_train_embed + t_test_embed),
    }

    table = Table(title=f"Linear probe -- {encoder_ckpt.parent.name}", show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("OOF (trainval)", justify="right")
    table.add_column("test", justify="right")
    for key in ("amex", "amex_g", "amex_d", "auc", "ks", "log_loss"):
        table.add_row(key, f"{float(oof_metrics[key]):.5f}", f"{float(test_metrics[key]):.5f}")
    console.print(table)

    # --- persist ---
    name = encoder_ckpt.parent.name  # e.g. "masked-8c94ccad"
    metrics_path = out_dir / f"{name}.json"
    oof_path = out_dir / f"{name}_oof.parquet"
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pl.DataFrame(
        {KEY_COL: trainval_id_list, "prediction": oof_pred, TARGET_COL: y_tv, FOLD_COL: fold_tv}
    ).write_parquet(oof_path, compression="zstd")
    console.print(f"[bold green]wrote {metrics_path} + {oof_path}[/]")


if __name__ == "__main__":
    app()
