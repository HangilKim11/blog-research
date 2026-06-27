"""Pretrain an SSL encoder under one of the four objectives via Hydra+Lightning.

CLI
---
    uv run python -m amex.ssl.pretrain --config-name masked
    uv run python -m amex.ssl.pretrain --config-name nextstep trainer.max_epochs=20
    uv run python -m amex.ssl.pretrain --config-name contrastive
    uv run python -m amex.ssl.pretrain --config-name hybrid

Outputs
-------
- checkpoint:  checkpoints/{objective}-{config_hash[:8]}/best.ckpt
- W&B run:     project=amex-ssl, tags=[ssl-pretrain, {objective}]
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import hydra
import polars as pl
import pytorch_lightning as pl_pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader

from amex.models.transformer import EncoderConfig
from amex.ssl.dataset import AmexSSLDataset, collate_ssl_samples
from amex.ssl.lightning_module import SSLPretrainModule
from amex.ssl.objectives import (
    ContrastiveConfig,
    ContrastiveSSL,
    HybridConfig,
    HybridMaskedContrastive,
    MaskedConfig,
    MaskedFeatureModeling,
    NextStepPrediction,
)
from amex.ssl.tokenizer import TokenizerArtifact

console = Console()

KEY_COL = "customer_ID"
SPLIT_COL = "split"


def _build_objective(cfg: DictConfig, tokenizer: TokenizerArtifact) -> nn.Module:
    """Dispatch to the requested objective module."""
    enc_cfg = EncoderConfig(**cfg.encoder)
    name = cfg.objective.name
    if name == "masked":
        return MaskedFeatureModeling(tokenizer, enc_cfg, MaskedConfig(**cfg.objective.params))
    if name == "nextstep":
        return NextStepPrediction(tokenizer, enc_cfg)
    if name == "contrastive":
        return ContrastiveSSL(tokenizer, enc_cfg, ContrastiveConfig(**cfg.objective.params))
    if name == "hybrid":
        return HybridMaskedContrastive(
            tokenizer,
            enc_cfg,
            masked_cfg=MaskedConfig(**cfg.objective.masked),
            contrastive_cfg=ContrastiveConfig(**cfg.objective.contrastive),
            cfg=HybridConfig(**cfg.objective.hybrid),
        )
    msg = f"unknown objective: {name}"
    raise ValueError(msg)


def _config_hash(cfg: DictConfig) -> str:
    """8-char hex digest of the resolved config; used in checkpoint paths."""
    serialized = json.dumps(OmegaConf.to_container(cfg, resolve=True), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]


def _make_loaders(
    cfg: DictConfig, tokenizer: TokenizerArtifact
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build the train + val DataLoaders by reading the canonical splits."""
    splits = pl.read_parquet(cfg.data.splits_path).select([KEY_COL, SPLIT_COL])
    train_ids = set(splits.filter(pl.col(SPLIT_COL) == "train")[KEY_COL].to_list())
    val_ids = set(splits.filter(pl.col(SPLIT_COL) == "val")[KEY_COL].to_list())
    console.print(f"[bold]splits[/] train={len(train_ids):,} val={len(val_ids):,}")

    train_ds = AmexSSLDataset(
        partition_glob=cfg.data.train_glob,
        tokenizer=tokenizer,
        customer_id_filter=train_ids,
        shuffle_partitions=True,
        shuffle_within_partition=True,
    )
    val_ds = AmexSSLDataset(
        partition_glob=cfg.data.train_glob,  # val customers also live in train partitions
        tokenizer=tokenizer,
        customer_id_filter=val_ids,
        shuffle_partitions=False,
        shuffle_within_partition=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_ssl_samples,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_ssl_samples,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    return train_loader, val_loader


@hydra.main(config_path="../../../configs/ssl", config_name="masked", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point; Hydra resolves the config first."""
    console.print(OmegaConf.to_yaml(cfg))

    # Optional seed override via Hydra (cfg.seed). Default 42 if missing.
    # Backward compat: if seed wasn't in the resolved config we keep the old
    # run-name format so existing checkpoints stay reachable by their hash.
    has_seed = "seed" in cfg
    seed = int(cfg.get("seed", 42))
    pl_pl.seed_everything(seed, workers=True)

    tokenizer = TokenizerArtifact.load(Path(cfg.data.tokenizer_path))
    cfg_hash = _config_hash(cfg)
    run_name = (
        f"{cfg.objective.name}-{cfg_hash}-s{seed}"
        if has_seed
        else f"{cfg.objective.name}-{cfg_hash}"
    )
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # also drop the resolved config alongside, for reproducibility
    (ckpt_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    objective = _build_objective(cfg, tokenizer)
    n_params = sum(p.numel() for p in objective.parameters() if p.requires_grad)
    console.print(f"[bold green]{run_name}[/] -- {n_params:,} trainable params")

    train_loader, val_loader = _make_loaders(cfg, tokenizer)

    module = SSLPretrainModule(
        objective=objective,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        max_steps=cfg.trainer.max_epochs * cfg.optim.steps_per_epoch_estimate,
        warmup_steps=cfg.optim.warmup_steps,
    )

    wandb_logger = (
        WandbLogger(
            project="amex-ssl",
            name=run_name,
            tags=["ssl-pretrain", cfg.objective.name],
            config=OmegaConf.to_container(cfg, resolve=True),
            save_dir=str(ckpt_dir),
        )
        if cfg.trainer.wandb
        else None
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(monitor="val/loss", mode="min", patience=cfg.trainer.patience),
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
        limit_val_batches=cfg.trainer.limit_val_batches,
        val_check_interval=cfg.trainer.val_check_interval,
    )
    trainer.fit(module, train_loader, val_loader)

    # Save the (encoder-only) state_dict separately so downstream linear-probe
    # doesn't need the objective head weights.
    encoder_path = ckpt_dir / "encoder.pt"
    import torch  # local import; the import-budget is fine here.

    # the objective always exposes a `.encoder` submodule (SequenceEncoder),
    # but mypy sees nn.Module's attribute as Tensor | Module union.
    encoder_module = cast("nn.Module", objective.encoder)
    torch.save(
        {
            "encoder_state_dict": encoder_module.state_dict(),
            "encoder_cfg": dict(cfg.encoder),
            "objective_name": cfg.objective.name,
            "config_hash": cfg_hash,
        },
        encoder_path,
    )
    console.print(f"[bold green]encoder saved:[/] {encoder_path}")


if __name__ == "__main__":
    main()
