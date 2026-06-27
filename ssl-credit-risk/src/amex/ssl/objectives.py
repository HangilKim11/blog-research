"""Four SSL objectives that share the same encoder backbone.

Each objective is a ``nn.Module`` that wraps an encoder plus the head(s)
it needs. ``forward(batch) -> dict`` returns at least ``{"loss": Tensor}``
plus per-component diagnostics (matches the CLAUDE.md rule
"All SSL objectives return both loss and a diagnostics dict").

Objectives implemented
----------------------
- :class:`MaskedFeatureModeling`   (BERT-style)
- :class:`NextStepPrediction`      (GPT-style, causal)
- :class:`ContrastiveSSL`          (SimCLR-style, two augmented views)
- :class:`HybridMaskedContrastive` (masked + contrastive jointly)

Each one runs end-to-end on a dict batch from ``collate_ssl_samples``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812 -- conventional PyTorch alias
from torch import Tensor, nn

from amex.models.transformer import (
    ContrastiveHead,
    EncoderConfig,
    ReconstructionHead,
    SequenceEncoder,
)
from amex.ssl.tokenizer import TokenizerArtifact


# ---------------------------------------------------------------------- #
# Configs
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class MaskedConfig:
    """Hyperparameters for the masked-feature objective."""

    mask_ratio: float = 0.15  # fraction of (timestep, feature) cells to mask
    num_loss_weight: float = 1.0
    cat_loss_weight: float = 1.0


@dataclass(frozen=True)
class ContrastiveConfig:
    """Hyperparameters for the SimCLR-style objective."""

    temperature: float = 0.1
    proj_dim: int = 128
    feature_dropout_p: float = 0.15
    min_crop_len: int = 6  # minimum surviving valid timesteps per view


@dataclass(frozen=True)
class HybridConfig:
    """Weighting between masked and contrastive losses in the hybrid."""

    mask_weight: float = 1.0
    contrastive_weight: float = 1.0


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _apply_masking(
    batch: dict[str, Tensor],
    mask_ratio: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Randomly mask (timestep, feature) cells.

    Returns ``(masked_batch, num_mask_pos, cat_mask_pos)`` where ``num_mask_pos``
    is (B, T, F) bool True at positions we masked AND that have a valid label
    (non-NaN, in-attention). Similar for ``cat_mask_pos`` (B, T, C).
    """
    B, T, F_num = batch["numeric_values"].shape
    C = batch["categorical_ids"].shape[-1]
    attn = batch["attention_mask"]  # (B, T)

    # Bernoulli mask on every cell.
    rng = torch.rand((B, T, F_num), device=attn.device, generator=generator)
    rng_cat = torch.rand((B, T, C), device=attn.device, generator=generator)
    num_mask_sample = rng < mask_ratio
    cat_mask_sample = rng_cat < mask_ratio

    # Only count loss where the position is in-attention and the original was a
    # valid (non-missing) label.
    valid_t = attn.unsqueeze(-1)  # (B, T, 1)
    num_valid_label = valid_t & ~batch["numeric_mask"]  # (B, T, F)
    cat_valid_label = valid_t & (batch["categorical_ids"] != 0)  # (B, T, C)

    num_mask_pos = num_mask_sample & num_valid_label
    cat_mask_pos = cat_mask_sample & cat_valid_label

    # Build the corrupted inputs (zero out the masked cells, mark numerics as missing).
    masked_num_vals = batch["numeric_values"].clone()
    masked_num_vals[num_mask_pos] = 0.0
    masked_num_mask = batch["numeric_mask"] | num_mask_pos
    masked_cat_ids = batch["categorical_ids"].clone()
    masked_cat_ids[cat_mask_pos] = 0

    new_batch = {
        **batch,
        "numeric_values": masked_num_vals,
        "numeric_mask": masked_num_mask,
        "categorical_ids": masked_cat_ids,
    }
    return new_batch, num_mask_pos, cat_mask_pos


def _reconstruction_loss(
    num_pred: Tensor,  # (B, T, F)
    cat_logits: list[Tensor],  # list of (B, T, V_i)
    num_target: Tensor,  # (B, T, F)
    cat_target: Tensor,  # (B, T, C)
    num_loss_mask: Tensor,  # (B, T, F) bool: contribute to loss?
    cat_loss_mask: Tensor,  # (B, T, C) bool
    *,
    num_weight: float = 1.0,
    cat_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Combined MSE + CE reconstruction loss with per-column averaging."""
    # numeric MSE
    num_count = num_loss_mask.sum().clamp_min(1).to(num_pred.dtype)
    num_loss = ((num_pred - num_target) ** 2 * num_loss_mask).sum() / num_count

    # categorical CE: one (B*T*1) -> (V_i) per head
    cat_losses: list[Tensor] = []
    for i, logits in enumerate(cat_logits):
        m = cat_loss_mask[..., i]  # (B, T) bool
        if m.any():
            logits_flat = logits[m]  # (N, V_i)
            target_flat = cat_target[..., i][m]  # (N,)
            cat_losses.append(F.cross_entropy(logits_flat, target_flat))
    cat_loss = torch.stack(cat_losses).mean() if cat_losses else num_pred.new_zeros(())

    total = num_weight * num_loss + cat_weight * cat_loss
    diag = {
        "loss/num_mse": float(num_loss.detach().item()),
        "loss/cat_ce": float(cat_loss.detach().item()) if cat_losses else 0.0,
        "mask/num_cells": float(num_loss_mask.sum().item()),
        "mask/cat_cells": float(cat_loss_mask.sum().item()),
    }
    return total, diag


def _info_nce(
    z_a: Tensor,  # (B, D) L2-normalized
    z_b: Tensor,  # (B, D) L2-normalized
    temperature: float,
) -> tuple[Tensor, dict[str, float]]:
    """Symmetric SimCLR InfoNCE loss for two views of the same B samples."""
    B = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)  # (2B, D)
    logits = (z @ z.t()) / temperature  # (2B, 2B)
    # mask out the diagonal (each item with itself) by setting to -inf
    diag_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(diag_mask, float("-inf"))
    # ground-truth: anchor i in view A pairs with i in view B (index B+i),
    # and anchor B+i in view B pairs with i in view A.
    targets = torch.cat(
        [torch.arange(B, device=z.device) + B, torch.arange(B, device=z.device)], dim=0
    )
    loss = F.cross_entropy(logits, targets)

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        acc = (pred == targets).float().mean().item()
    return loss, {"loss/contrastive": float(loss.detach().item()), "contrast/top1": acc}


def _augment_for_contrastive(
    batch: dict[str, Tensor],
    cfg: ContrastiveConfig,
) -> dict[str, Tensor]:
    """Build one augmented view: random temporal crop + feature dropout."""
    B, T, F_num = batch["numeric_values"].shape
    C = batch["categorical_ids"].shape[-1]
    attn = batch["attention_mask"]  # (B, T)
    device = attn.device

    # --- temporal crop: per-sample, choose a contiguous keep window of size L
    # in [min_crop_len, seq_len], and randomly position it within the valid prefix.
    seq_len = attn.sum(dim=1)  # (B,)
    new_attn = attn.clone()
    for i in range(B):
        n = int(seq_len[i].item())
        if n <= cfg.min_crop_len:
            continue
        keep_len = int(torch.randint(cfg.min_crop_len, n + 1, (1,), device=device).item())
        start = int(torch.randint(0, n - keep_len + 1, (1,), device=device).item())
        new_row = torch.zeros(T, dtype=attn.dtype, device=device)
        new_row[start : start + keep_len] = True
        new_attn[i] = new_row

    # --- feature dropout: zero out random columns per-sample on numerics; for
    # categoricals, set to MISSING (code 0).
    num_drop = torch.rand((B, F_num), device=device) < cfg.feature_dropout_p  # (B, F)
    cat_drop = torch.rand((B, C), device=device) < cfg.feature_dropout_p  # (B, C)

    aug_num_vals = batch["numeric_values"].clone()
    aug_num_vals[num_drop.unsqueeze(1).expand(-1, T, -1)] = 0.0
    aug_num_mask = batch["numeric_mask"] | num_drop.unsqueeze(1).expand(-1, T, -1)

    aug_cat_ids = batch["categorical_ids"].clone()
    aug_cat_ids[cat_drop.unsqueeze(1).expand(-1, T, -1)] = 0

    return {
        **batch,
        "numeric_values": aug_num_vals,
        "numeric_mask": aug_num_mask,
        "categorical_ids": aug_cat_ids,
        "attention_mask": new_attn,
    }


# ---------------------------------------------------------------------- #
# Objective modules
# ---------------------------------------------------------------------- #
class MaskedFeatureModeling(nn.Module):
    """BERT-style: mask 15% of cells, reconstruct the originals."""

    def __init__(
        self,
        tokenizer: TokenizerArtifact,
        enc_cfg: EncoderConfig,
        cfg: MaskedConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or MaskedConfig()
        self.tokenizer = tokenizer
        self.encoder = SequenceEncoder(tokenizer, enc_cfg)
        self.head = ReconstructionHead(tokenizer, d_model=enc_cfg.d_model)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
        num_target = batch["numeric_values"]
        cat_target = batch["categorical_ids"]
        corrupted, num_mask_pos, cat_mask_pos = _apply_masking(batch, self.cfg.mask_ratio)

        hidden = self.encoder(
            corrupted["numeric_values"],
            corrupted["numeric_mask"],
            corrupted["categorical_ids"],
            corrupted["attention_mask"],
        )  # (B, T+1, D) with CLS at index 0
        per_t = hidden[:, 1:, :] if self.encoder.cfg.use_cls_token else hidden

        num_pred, cat_logits = self.head(per_t)
        loss, diag = _reconstruction_loss(
            num_pred,
            cat_logits,
            num_target,
            cat_target,
            num_loss_mask=num_mask_pos,
            cat_loss_mask=cat_mask_pos,
            num_weight=self.cfg.num_loss_weight,
            cat_weight=self.cfg.cat_loss_weight,
        )
        return {"loss": loss, **diag}


class NextStepPrediction(nn.Module):
    """GPT-style: causal attention, predict feature vector at t+1 from <=t."""

    def __init__(
        self,
        tokenizer: TokenizerArtifact,
        enc_cfg: EncoderConfig,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        # Causal next-step pretraining shouldn't add a CLS token (cleaner causal flow).
        enc_cfg_nocls = EncoderConfig(
            d_model=enc_cfg.d_model,
            n_layers=enc_cfg.n_layers,
            n_heads=enc_cfg.n_heads,
            dim_feedforward=enc_cfg.dim_feedforward,
            dropout=enc_cfg.dropout,
            cat_embed_dim=enc_cfg.cat_embed_dim,
            max_seq_len=enc_cfg.max_seq_len,
            use_cls_token=False,
        )
        self.encoder = SequenceEncoder(tokenizer, enc_cfg_nocls)
        self.head = ReconstructionHead(tokenizer, d_model=enc_cfg.d_model)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
        hidden = self.encoder(
            batch["numeric_values"],
            batch["numeric_mask"],
            batch["categorical_ids"],
            batch["attention_mask"],
            causal=True,
        )  # (B, T, D)
        num_pred, cat_logits = self.head(hidden)  # predicts the position contents

        # Targets are shifted by +1: predict position t+1 from output at position t.
        # We keep predictions for positions [0..T-2], compare with targets at [1..T-1].
        num_pred_shift = num_pred[:, :-1, :]  # (B, T-1, F)
        num_tgt_shift = batch["numeric_values"][:, 1:, :]  # (B, T-1, F)
        cat_logits_shift = [c[:, :-1, :] for c in cat_logits]  # each (B, T-1, V_i)
        cat_tgt_shift = batch["categorical_ids"][:, 1:, :]  # (B, T-1, C)

        # Loss mask: predict only where BOTH t and t+1 are in-attention.
        attn = batch["attention_mask"]
        pair_attn = attn[:, :-1] & attn[:, 1:]  # (B, T-1)
        num_loss_mask = pair_attn.unsqueeze(-1) & ~batch["numeric_mask"][:, 1:, :]
        cat_loss_mask = pair_attn.unsqueeze(-1) & (batch["categorical_ids"][:, 1:, :] != 0)

        loss, diag = _reconstruction_loss(
            num_pred_shift,
            cat_logits_shift,
            num_tgt_shift,
            cat_tgt_shift,
            num_loss_mask=num_loss_mask,
            cat_loss_mask=cat_loss_mask,
        )
        return {"loss": loss, **diag}


class ContrastiveSSL(nn.Module):
    """SimCLR-style two-view InfoNCE on CLS embeddings."""

    def __init__(
        self,
        tokenizer: TokenizerArtifact,
        enc_cfg: EncoderConfig,
        cfg: ContrastiveConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ContrastiveConfig()
        self.tokenizer = tokenizer
        # CLS is required for the global representation.
        if not enc_cfg.use_cls_token:
            msg = "ContrastiveSSL requires use_cls_token=True"
            raise ValueError(msg)
        self.encoder = SequenceEncoder(tokenizer, enc_cfg)
        self.proj = ContrastiveHead(d_model=enc_cfg.d_model, proj_dim=self.cfg.proj_dim)

    def _encode_view(self, view: dict[str, Tensor]) -> Tensor:
        h = self.encoder(
            view["numeric_values"],
            view["numeric_mask"],
            view["categorical_ids"],
            view["attention_mask"],
        )  # (B, T+1, D), CLS at 0
        return self.proj(h[:, 0, :])  # (B, proj_dim)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
        view_a = _augment_for_contrastive(batch, self.cfg)
        view_b = _augment_for_contrastive(batch, self.cfg)
        z_a = self._encode_view(view_a)
        z_b = self._encode_view(view_b)
        loss, diag = _info_nce(z_a, z_b, self.cfg.temperature)
        return {"loss": loss, **diag}


class HybridMaskedContrastive(nn.Module):
    """Joint loss: alpha * MaskedFeatureModeling + beta * ContrastiveSSL.

    The two heads share the same encoder; we run two forwards per step (one
    masked, one contrastive). This is the simplest "hybrid" combination and
    the one the user picked from the menu.
    """

    def __init__(
        self,
        tokenizer: TokenizerArtifact,
        enc_cfg: EncoderConfig,
        masked_cfg: MaskedConfig | None = None,
        contrastive_cfg: ContrastiveConfig | None = None,
        cfg: HybridConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or HybridConfig()
        if not enc_cfg.use_cls_token:
            msg = "HybridMaskedContrastive requires use_cls_token=True"
            raise ValueError(msg)
        self.tokenizer = tokenizer
        self.encoder = SequenceEncoder(tokenizer, enc_cfg)
        self.recon_head = ReconstructionHead(tokenizer, d_model=enc_cfg.d_model)
        self.proj_head = ContrastiveHead(
            d_model=enc_cfg.d_model,
            proj_dim=(contrastive_cfg or ContrastiveConfig()).proj_dim,
        )
        self.masked_cfg = masked_cfg or MaskedConfig()
        self.contrastive_cfg = contrastive_cfg or ContrastiveConfig()

    def _encode(self, view: dict[str, Tensor]) -> Tensor:
        return self.encoder(
            view["numeric_values"],
            view["numeric_mask"],
            view["categorical_ids"],
            view["attention_mask"],
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
        # --- masked branch ---
        num_target = batch["numeric_values"]
        cat_target = batch["categorical_ids"]
        corrupted, num_mask_pos, cat_mask_pos = _apply_masking(batch, self.masked_cfg.mask_ratio)
        h_m = self._encode(corrupted)
        per_t = h_m[:, 1:, :]
        num_pred, cat_logits = self.recon_head(per_t)
        mask_loss, mask_diag = _reconstruction_loss(
            num_pred,
            cat_logits,
            num_target,
            cat_target,
            num_loss_mask=num_mask_pos,
            cat_loss_mask=cat_mask_pos,
            num_weight=self.masked_cfg.num_loss_weight,
            cat_weight=self.masked_cfg.cat_loss_weight,
        )

        # --- contrastive branch ---
        view_a = _augment_for_contrastive(batch, self.contrastive_cfg)
        view_b = _augment_for_contrastive(batch, self.contrastive_cfg)
        z_a = self.proj_head(self._encode(view_a)[:, 0, :])
        z_b = self.proj_head(self._encode(view_b)[:, 0, :])
        contr_loss, contr_diag = _info_nce(z_a, z_b, self.contrastive_cfg.temperature)

        total = self.cfg.mask_weight * mask_loss + self.cfg.contrastive_weight * contr_loss
        return {
            "loss": total,
            "loss/mask_total": float(mask_loss.detach().item()),
            "loss/contrastive_total": float(contr_loss.detach().item()),
            **mask_diag,
            **contr_diag,
        }
