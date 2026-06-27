"""End-to-end fine-tune of a pretrained SSL encoder for AMEX default.

Loads a frozen-from-disk encoder, attaches a fresh ``ClassificationHead``,
and trains the whole stack (encoder + head) jointly with BCE loss.
Layer-wise lr decay puts a small lr on the encoder and a larger lr on the
head, which preserves the pretrained weights while letting the new head
calibrate quickly.

Validation tracks the AMEX competition metric directly; EarlyStopping +
ModelCheckpoint both watch ``val/amex`` (mode=max).

CLI
---
    uv run python -m amex.finetune.full_finetune \\
        --config-name default encoder=checkpoints/nextstep-274fdf8c/encoder.pt
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import polars as pl
import pytorch_lightning as pl_pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from rich.console import Console
from rich.table import Table
from sklearn.metrics import log_loss, roc_auc_score
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, IterableDataset

from amex.evaluation.metrics import amex_metric_components
from amex.models.transformer import ClassificationHead, EncoderConfig, SequenceEncoder
from amex.ssl.dataset import AmexSSLDataset, collate_ssl_samples
from amex.ssl.tokenizer import TokenizerArtifact

console = Console()

KEY_COL = "customer_ID"
TARGET_COL = "target"
SPLIT_COL = "split"
TEST_SPLIT = "test"


# ---------------------------------------------------------------------- #
# Dataset wrapper that adds a target field to each SSL sample.
# ---------------------------------------------------------------------- #
class AmexLabeledDataset(IterableDataset[dict[str, Any]]):
    """Wraps AmexSSLDataset, joins per-customer target labels onto each sample."""

    def __init__(self, ssl_dataset: AmexSSLDataset, target_map: dict[str, int]) -> None:
        super().__init__()
        self.ssl_dataset = ssl_dataset
        self.target_map = target_map

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for sample in self.ssl_dataset:
            cid = sample["customer_id"]
            if cid not in self.target_map:
                continue  # skip customers without labels
            sample["target"] = self.target_map[cid]
            yield sample


def collate_labeled(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_ssl_samples(samples)
    out["target"] = torch.tensor([s["target"] for s in samples], dtype=torch.float32)
    return out


# ---------------------------------------------------------------------- #
# Encoder + classification head, packaged as one nn.Module.
# ---------------------------------------------------------------------- #
class EncoderWithHead(nn.Module):
    """Forward returns logits (B,). Pools sequence into a single vector."""

    def __init__(self, encoder: SequenceEncoder, head: ClassificationHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        hidden = self.encoder(
            batch["numeric_values"],
            batch["numeric_mask"],
            batch["categorical_ids"],
            batch["attention_mask"],
        )  # (B, T', D)
        if self.encoder.cfg.use_cls_token:
            pooled = hidden[:, 0, :]  # CLS
        else:
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        return self.head(pooled)  # (B,) logits


# ---------------------------------------------------------------------- #
# Lightning module
# ---------------------------------------------------------------------- #
class FinetuneModule(pl_pl.LightningModule):
    """Joint fine-tune: BCE loss, AMEX metric for early stopping."""

    def __init__(
        self,
        model: EncoderWithHead,
        lr_head: float = 1e-3,
        lr_encoder: float = 1e-4,
        weight_decay: float = 0.01,
        max_steps: int = 10_000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.lr_head = lr_head
        self.lr_encoder = lr_encoder
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        # buffers for val/test metric computation
        self._val_pred: list[np.ndarray] = []
        self._val_true: list[np.ndarray] = []

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        logits = self.model(batch)
        loss = self.loss_fn(logits, batch["target"])
        bs = int(batch["target"].shape[0])
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=bs, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_pred.clear()
        self._val_true.clear()

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        logits = self.model(batch)
        loss = self.loss_fn(logits, batch["target"])
        bs = int(batch["target"].shape[0])
        self.log("val/loss", loss, on_epoch=True, batch_size=bs)
        # accumulate for AMEX
        with torch.no_grad():
            self._val_pred.append(torch.sigmoid(logits).float().cpu().numpy())
            self._val_true.append(batch["target"].float().cpu().numpy())
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_pred:
            return
        y_pred = np.concatenate(self._val_pred, axis=0)
        y_true = np.concatenate(self._val_true, axis=0)
        m, g, d = amex_metric_components(y_true, y_pred)
        self.log("val/amex", m, prog_bar=True)
        self.log("val/amex_g", g)
        self.log("val/amex_d", d)
        if len(np.unique(y_true)) > 1:
            self.log("val/auc", float(roc_auc_score(y_true, y_pred)))

    # -------- optim with two lr groups -------- #
    def configure_optimizers(self) -> Any:
        encoder_params = list(self.model.encoder.parameters())
        head_params = list(self.model.head.parameters())
        opt = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": self.lr_encoder},
                {"params": head_params, "lr": self.lr_head},
            ],
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        sched = CosineAnnealingLR(opt, T_max=self.max_steps)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _load_encoder(ckpt_path: Path, tokenizer: TokenizerArtifact) -> SequenceEncoder:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_cfg = EncoderConfig(**payload["encoder_cfg"])
    enc = SequenceEncoder(tokenizer, enc_cfg)
    enc.load_state_dict(payload["encoder_state_dict"])
    return enc


def _build_loaders(
    cfg: DictConfig, tokenizer: TokenizerArtifact, splits: pl.DataFrame
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """train (=train+val [or subset]), val (=val split early-stop proxy), test."""
    train_ids = set(splits.filter(pl.col(SPLIT_COL).is_in(["train", "val"]))[KEY_COL].to_list())
    earlystop_ids = set(splits.filter(pl.col(SPLIT_COL) == "val")[KEY_COL].to_list())
    test_ids = set(splits.filter(pl.col(SPLIT_COL) == TEST_SPLIT)[KEY_COL].to_list())

    # Few-shot study hook: subsample the labeled trainval pool. We carve a
    # separate 20% slice of the SUBSET to serve as the early-stop set, so the
    # few-shot model never sees labels outside its budget. (Naively
    # intersecting with the original val split can leave fewer rows than the
    # batch size, producing an empty val dataloader.)
    subset_fraction = float(cfg.data.get("subset_fraction", 1.0) or 1.0)
    if subset_fraction < 1.0:
        from amex.data.subset import stratified_trainval_subset

        subset_ids = stratified_trainval_subset(
            Path(cfg.data.splits_path),
            subset_fraction,
            seed=int(cfg.data.get("subset_seed", 1234)),
        )
        subset_list = sorted(subset_ids)
        # Deterministic shuffle with a *different* seed than subset selection.
        np.random.default_rng(int(cfg.data.get("subset_seed", 1234)) + 1).shuffle(subset_list)
        n_val = max(1, int(len(subset_list) * 0.2))
        earlystop_ids = set(subset_list[:n_val])
        train_ids = set(subset_list[n_val:])
        console.print(
            f"[bold yellow]few-shot subset:[/] fraction={subset_fraction} "
            f"-> train={len(train_ids):,} / earlystop={len(earlystop_ids):,} "
            f"(carved from subset of {len(subset_list):,})"
        )
    target_map = {
        cid: int(t)
        for cid, t in zip(splits[KEY_COL].to_list(), splits[TARGET_COL].to_list(), strict=True)
    }
    console.print(
        f"[bold]splits[/] train+val={len(train_ids):,} earlystop(val)={len(earlystop_ids):,} test={len(test_ids):,}"
    )

    def _mk(filter_ids: set[str], *, shuffle: bool) -> DataLoader[Any]:
        ssl_ds = AmexSSLDataset(
            partition_glob=cfg.data.train_glob,
            tokenizer=tokenizer,
            customer_id_filter=filter_ids,
            shuffle_partitions=shuffle,
            shuffle_within_partition=shuffle,
        )
        labeled = AmexLabeledDataset(ssl_ds, target_map)
        return DataLoader(
            labeled,
            batch_size=cfg.data.batch_size,
            num_workers=0,
            collate_fn=collate_labeled,
            pin_memory=True,
        )

    return (
        _mk(train_ids, shuffle=True),
        _mk(earlystop_ids, shuffle=False),
        _mk(test_ids, shuffle=False),
    )


def _evaluate_test(
    module: FinetuneModule, loader: DataLoader[Any], device: str
) -> dict[str, float]:
    module.eval().to(device)
    preds: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            moved = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.autocast(
                device_type="cuda" if device == "cuda" else "cpu", dtype=torch.bfloat16
            ):
                logits = module.model(moved)
            preds.append(torch.sigmoid(logits).float().cpu().numpy())
            ys.append(moved["target"].float().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(ys, axis=0)
    m, g, d = amex_metric_components(y_true, y_pred)
    return {
        "amex": m,
        "amex_g": g,
        "amex_d": d,
        "auc": float(roc_auc_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, np.clip(y_pred, 1e-15, 1 - 1e-15))),
        "n": len(y_true),
    }


# ---------------------------------------------------------------------- #
# Main entry
# ---------------------------------------------------------------------- #
@hydra.main(config_path="../../../configs/finetune", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Fine-tune one pretrained encoder."""
    console.print(OmegaConf.to_yaml(cfg))

    tokenizer = TokenizerArtifact.load(Path(cfg.data.tokenizer_path))
    splits = pl.read_parquet(cfg.data.splits_path).select([KEY_COL, TARGET_COL, SPLIT_COL])

    encoder_ckpt = Path(cfg.encoder.ckpt)
    encoder = _load_encoder(encoder_ckpt, tokenizer)
    head = ClassificationHead(d_model=encoder.cfg.d_model, dropout=cfg.head.dropout)
    model = EncoderWithHead(encoder, head)
    n_params = sum(p.numel() for p in model.parameters())
    console.print(f"[bold green]model[/] {n_params:,} params (encoder + head)")

    train_loader, val_loader, test_loader = _build_loaders(cfg, tokenizer, splits)
    steps_per_epoch = cfg.trainer.steps_per_epoch_estimate
    module = FinetuneModule(
        model=model,
        lr_head=cfg.optim.lr_head,
        lr_encoder=cfg.optim.lr_encoder,
        weight_decay=cfg.optim.weight_decay,
        max_steps=cfg.trainer.max_epochs * steps_per_epoch,
    )

    run_name = f"finetune-{encoder_ckpt.parent.name}"
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    wandb_logger = (
        WandbLogger(
            project="amex-ssl",
            name=run_name,
            tags=["finetune", encoder_ckpt.parent.name.split("-")[0]],
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=str(ckpt_dir),
        )
        if cfg.trainer.wandb
        else None
    )
    # EarlyStopping is `strict=False` so we don't crash if validation is skipped
    # (Lightning + IterableDataset + train-data exhaustion before
    # limit_train_batches will silently NOT trigger val_check_interval=1.0).
    # ModelCheckpoint saves both best (when val/amex is available) and last
    # so we always have a checkpoint to load.
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor="val/amex",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/amex",
            mode="max",
            patience=cfg.trainer.patience,
            strict=False,
        ),
    ]
    trainer = pl_pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.optim.grad_clip,
        limit_train_batches=cfg.trainer.limit_train_batches,
        num_sanity_val_steps=0,
    )
    t0 = time.monotonic()
    trainer.fit(module, train_loader, val_loader)
    train_seconds = time.monotonic() - t0

    # Reload best ckpt (val-amex-selected) if available, else last ckpt, else
    # the in-memory module (small fractions often skip val entirely).
    best_path = getattr(callbacks[0], "best_model_path", "") or ""
    last_path = getattr(callbacks[0], "last_model_path", "") or ""
    if best_path and Path(best_path).exists():
        console.print(f"[bold]loading best ckpt (val/amex):[/] {best_path}")
        best: FinetuneModule = FinetuneModule.load_from_checkpoint(best_path, model=model)
    elif last_path and Path(last_path).exists():
        console.print(f"[bold yellow]val/amex unavailable -- loading last ckpt:[/] {last_path}")
        best = FinetuneModule.load_from_checkpoint(last_path, model=model)
    else:
        console.print("[bold yellow]no ckpt on disk; using in-memory module[/]")
        best = module

    device = "cuda" if cfg.trainer.accelerator == "gpu" else "cpu"
    test_metrics = _evaluate_test(best, test_loader, device)
    summary = {
        "run_name": run_name,
        "encoder_ckpt": str(encoder_ckpt),
        "n_params": n_params,
        "best_ckpt": str(best_path),
        "train_seconds": train_seconds,
        "test": test_metrics,
    }
    out_path = Path("data/processed/v1/finetunes")
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_file = out_path / f"{run_name}.json"
    metrics_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    table = Table(title=f"Fine-tune -- {run_name}", show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("test", justify="right")
    for k, v in test_metrics.items():
        table.add_row(k, f"{v:.5f}" if isinstance(v, float) else str(v))
    console.print(table)
    console.print(f"[bold green]wrote {metrics_file}[/]")

    if wandb_logger is not None:
        wandb_logger.log_metrics(
            {f"test/{k}": float(v) for k, v in test_metrics.items() if isinstance(v, int | float)}
        )


if __name__ == "__main__":
    main()
