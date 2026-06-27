"""PyTorch Lightning wrapper around any SSL objective module.

A thin shell: we log every diagnostic the objective returns, run an
AdamW + cosine schedule, and emit one consolidated ``val/loss`` signal
that ``EarlyStopping`` and the checkpoint callback watch.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from amex.evaluation.metrics import amex_metric_components


class SSLPretrainModule(pl.LightningModule):
    """Wraps an SSL objective ``nn.Module`` for the Lightning Trainer."""

    def __init__(
        self,
        objective: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        max_steps: int = 100_000,
        warmup_steps: int = 500,
    ) -> None:
        super().__init__()
        # Don't save the objective itself in the hparams blob (it's not picklable
        # as a hparam) but do save scalars for reproducibility.
        self.save_hyperparameters(ignore=["objective"])
        self.objective = objective
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps

    # -------- training / val -------- #
    def _log_outputs(self, out: dict[str, Tensor | float], stage: str) -> None:
        for k, v in out.items():
            tag = f"{stage}/{k.split('/', 1)[-1]}" if "/" in k else f"{stage}/{k}"
            self.log(
                tag,
                float(v.item() if isinstance(v, torch.Tensor) else v),
                prog_bar=(k == "loss"),
                on_step=(stage == "train"),
                on_epoch=True,
                batch_size=self._infer_batch_size(),
            )

    def _infer_batch_size(self) -> int:
        # Lightning needs an explicit batch size for IterableDataset.
        try:
            loader = self.trainer.train_dataloader
            if loader is None:
                return 1
            return int(loader.batch_size or 1)
        except (AttributeError, RuntimeError):
            return 1

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        out = self.objective(batch)
        self._log_outputs(out, "train")
        return out["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        out = self.objective(batch)
        self._log_outputs(out, "val")
        return out["loss"]

    # -------- optim -------- #
    def configure_optimizers(self) -> Any:
        # AdamW with weight decay (skip biases + LayerNorm per common practice).
        decay, no_decay = [], []
        for n, p in self.objective.named_parameters():
            if not p.requires_grad:
                continue
            if (
                n.endswith(".bias")
                or "norm" in n.lower()
                or "pos_embedding" in n
                or "cls_token" in n
            ):
                no_decay.append(p)
            else:
                decay.append(p)
        opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=(0.9, 0.95),
        )
        sched = CosineAnnealingLR(opt, T_max=self.max_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


# ----------------------------------------------------------------------
# Helper used by linear-probe to compute AMEX metric outside the trainer.
# ----------------------------------------------------------------------
def amex_dict(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Returns {amex, amex_g, amex_d} for downstream eval."""
    m, g, d = amex_metric_components(y_true, y_pred)
    return {"amex": m, "amex_g": g, "amex_d": d}
