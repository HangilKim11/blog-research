"""Extract mean-pooled SSL encoder embeddings for every customer.

Output a single Parquet ``[customer_ID, emb_000..emb_127]`` (or whatever
``d_model`` the encoder uses), covering both train+val and test partitions
so downstream feature concatenation has full coverage.

Usage
-----
    uv run python -m amex.baseline.ssl_features --encoder checkpoints/hybrid-e3d5d881/encoder.pt
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import typer
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from amex.models.transformer import EncoderConfig, SequenceEncoder
from amex.ssl.dataset import AmexSSLDataset, collate_ssl_samples
from amex.ssl.tokenizer import TokenizerArtifact

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=False)

KEY_COL = "customer_ID"


def _load_encoder(ckpt_path: Path, tokenizer: TokenizerArtifact) -> SequenceEncoder:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_cfg = EncoderConfig(**payload["encoder_cfg"])
    enc = SequenceEncoder(tokenizer, enc_cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    enc.eval()
    return enc


def _embed_partitions(
    encoder: SequenceEncoder,
    tokenizer: TokenizerArtifact,
    partition_glob: str,
    *,
    batch_size: int,
    device: str,
) -> tuple[list[str], np.ndarray]:
    """Iterate a partition tree and return (customer_ids, embeddings) for all rows.

    No customer-id filter -- we want every customer in the tree.
    """
    ds = AmexSSLDataset(
        partition_glob=partition_glob,
        tokenizer=tokenizer,
        customer_id_filter=None,
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
    ids: list[str] = []
    feats: list[np.ndarray] = []
    use_cuda = device == "cuda"
    autocast_dtype = torch.bfloat16
    with torch.inference_mode():
        for batch in loader:
            num_vals = batch["numeric_values"].to(device, non_blocking=True)
            num_mask = batch["numeric_mask"].to(device, non_blocking=True)
            cat_ids = batch["categorical_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=autocast_dtype):
                hidden = encoder(num_vals, num_mask, cat_ids, attn)  # (B, T', D)
            per_t = hidden[:, 1:, :] if encoder.cfg.use_cls_token else hidden
            mask = attn.unsqueeze(-1).to(per_t.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (per_t * mask).sum(dim=1) / denom  # (B, D)
            feats.append(pooled.float().cpu().numpy())
            ids.extend(batch["customer_id"])
    return ids, np.concatenate(feats, axis=0)


@app.command()
def main(
    encoder_ckpt: Path = typer.Option(..., "--encoder", help="encoder.pt from ssl.pretrain."),
    tokenizer_path: Path = typer.Option(Path("data/processed/v1/tokenizer.json"), "--tokenizer"),
    train_glob: str = typer.Option("data/processed/v1/train/**/*.parquet", "--train-glob"),
    test_glob: str = typer.Option("data/processed/v1/test/**/*.parquet", "--test-glob"),
    out_path: Path = typer.Option(
        Path("data/processed/v1/features_ssl_hybrid.parquet"),
        "--out",
        help="destination parquet (customer_ID + emb_NNN columns).",
    ),
    batch_size: int = typer.Option(512, "--batch-size"),
    device: str = typer.Option("cuda", "--device"),
    prefix: str = typer.Option(
        "ssl_hybrid", "--prefix", help="column-name prefix for the emb cols."
    ),
) -> None:
    """Run the encoder over all customers and dump a 128-dim embedding parquet."""
    tokenizer = TokenizerArtifact.load(tokenizer_path)
    encoder = _load_encoder(encoder_ckpt, tokenizer)
    n_params = sum(p.numel() for p in encoder.parameters())
    d = encoder.cfg.d_model
    console.print(f"[bold]loaded encoder[/] {encoder_ckpt} ({n_params:,} params, d_model={d})")

    t0 = time.monotonic()
    tv_ids, tv_emb = _embed_partitions(
        encoder, tokenizer, train_glob, batch_size=batch_size, device=device
    )
    t_tv = time.monotonic() - t0
    console.print(f"  train+val: {tv_emb.shape} in {t_tv:,.1f}s")

    t0 = time.monotonic()
    te_ids, te_emb = _embed_partitions(
        encoder, tokenizer, test_glob, batch_size=batch_size, device=device
    )
    t_te = time.monotonic() - t0
    console.print(f"  test:      {te_emb.shape} in {t_te:,.1f}s")

    all_ids = tv_ids + te_ids
    all_emb = np.concatenate([tv_emb, te_emb], axis=0)
    if len(set(all_ids)) != len(all_ids):
        console.print(
            f"[red]duplicate customer ids detected:[/] {len(all_ids) - len(set(all_ids))} dups"
        )

    cols: dict[str, list[str] | np.ndarray] = {KEY_COL: all_ids}
    for j in range(d):
        cols[f"{prefix}_emb_{j:03d}"] = all_emb[:, j].astype(np.float32)
    frame = pl.DataFrame(cols).sort(KEY_COL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path, compression="zstd")

    table = Table(title="SSL feature parquet", show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("encoder", encoder_ckpt.parent.name)
    table.add_row("rows (customers)", f"{frame.height:,}")
    table.add_row("emb cols", f"{d}")
    table.add_row("file size", f"{out_path.stat().st_size / 1024**2:,.1f} MB")
    table.add_row("wall time", f"{t_tv + t_te:,.1f} s")
    console.print(table)
    console.print(f"[bold green]wrote[/] {out_path}")


if __name__ == "__main__":
    app()
