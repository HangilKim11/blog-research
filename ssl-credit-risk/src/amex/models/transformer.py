"""Shared transformer encoder + objective-specific heads for SSL pretraining.

The same encoder backbone is used by every SSL objective (masked feature
modeling, next-step prediction, contrastive, hybrid) and by every downstream
task (linear probe, full fine-tune). Only the heads change.

Tensor shape glossary
---------------------
- B  = batch size
- T  = sequence length (= MAX_SEQ_LEN = 13)
- F  = number of numeric features (177 on the real data)
- C  = number of categorical columns (11 on the real data)
- D  = model hidden dim (default 128)
- V_i = vocab size of categorical column i (3..8 on the real data)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from amex.ssl.tokenizer import TokenizerArtifact


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EncoderConfig:
    """Hyperparameters for the shared encoder backbone."""

    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    cat_embed_dim: int = 8
    max_seq_len: int = 13  # matches tokenizer.MAX_SEQ_LEN
    use_cls_token: bool = True


# ----------------------------------------------------------------------
# Feature embedder
# ----------------------------------------------------------------------
class FeatureEmbedder(nn.Module):
    """Embed (numeric_values, numeric_mask, categorical_ids) -> (B, T, D).

    - Numeric: stacks values with a 0/1 missing-indicator (so F effective = 2*F),
      then a single linear projects to D.
    - Categorical: per-column nn.Embedding, concatenated, then linear to D.
    - Output: numeric_proj + categorical_proj (sum -> 1 vector per timestep).
    """

    def __init__(self, tokenizer: TokenizerArtifact, cfg: EncoderConfig) -> None:
        super().__init__()
        self.n_num = tokenizer.n_numeric
        self.n_cat = tokenizer.n_categorical
        self.d_model = cfg.d_model

        # Numeric: concat(value, mask_as_float) -> Linear -> D
        self.numeric_proj = nn.Linear(2 * self.n_num, cfg.d_model)

        # Categorical: per-column embedding tables, padding_idx=0 for MISSING
        self.cat_embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=tokenizer.cat_vocab_size(col),
                    embedding_dim=cfg.cat_embed_dim,
                    padding_idx=0,
                )
                for col in tokenizer.categorical_cols
            ]
        )
        self.cat_proj = nn.Linear(self.n_cat * cfg.cat_embed_dim, cfg.d_model)

    def forward(
        self,
        numeric_values: Tensor,  # (B, T, F)
        numeric_mask: Tensor,  # (B, T, F) bool
        categorical_ids: Tensor,  # (B, T, C) int64
    ) -> Tensor:  # (B, T, D)
        # numeric: stack value + mask_as_float along the feature dim
        mask_f = numeric_mask.to(numeric_values.dtype)
        num_in = torch.cat([numeric_values, mask_f], dim=-1)  # (B, T, 2F)
        num_emb = self.numeric_proj(num_in)  # (B, T, D)

        # categorical: per-column embed, concat, project
        # categorical_ids: (B, T, C); each column slice -> (B, T) ints -> (B, T, embed_dim)
        cat_embs = [
            emb(categorical_ids[..., i])  # (B, T, embed_dim)
            for i, emb in enumerate(self.cat_embeddings)
        ]
        cat_concat = torch.cat(cat_embs, dim=-1)  # (B, T, C*embed_dim)
        cat_emb = self.cat_proj(cat_concat)  # (B, T, D)

        return num_emb + cat_emb  # (B, T, D)


# ----------------------------------------------------------------------
# Encoder backbone
# ----------------------------------------------------------------------
class SequenceEncoder(nn.Module):
    """Feature embed + learned pos enc + (optional CLS) + transformer stack.

    Inputs (batched): numeric_values (B,T,F), numeric_mask (B,T,F),
    categorical_ids (B,T,C), attention_mask (B,T) bool.

    Output: token-wise embeddings (B, T', D) where T' = T+1 if use_cls_token
    else T. The CLS position is index 0 when present.
    """

    def __init__(self, tokenizer: TokenizerArtifact, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedder = FeatureEmbedder(tokenizer, cfg)

        # Learned positional encoding (T+1 slots if CLS, else T).
        eff_len = cfg.max_seq_len + (1 if cfg.use_cls_token else 0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, eff_len, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        if cfg.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer stack, Pre-LN for SSL stability.
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        numeric_values: Tensor,  # (B, T, F)
        numeric_mask: Tensor,  # (B, T, F) bool
        categorical_ids: Tensor,  # (B, T, C) int64
        attention_mask: Tensor,  # (B, T) bool, True = valid
        *,
        causal: bool = False,
    ) -> Tensor:  # (B, T', D)
        x = self.embedder(numeric_values, numeric_mask, categorical_ids)  # (B, T, D)
        B = x.size(0)

        if self.cfg.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            x = torch.cat([cls, x], dim=1)  # (B, T+1, D)
            cls_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=x.device)
            attention_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, T+1)

        x = x + self.pos_embedding[:, : x.size(1)]  # broadcast pos enc

        # PyTorch transformer wants src_key_padding_mask = True for *padding*.
        src_key_padding_mask = ~attention_mask  # True where we want to mask out

        attn_mask: Tensor | None = None
        if causal:
            # nn.Transformer wants a (T', T') float mask with -inf where blocked.
            Tp = x.size(1)
            attn_mask = torch.triu(torch.full((Tp, Tp), float("-inf"), device=x.device), diagonal=1)

        out = self.transformer(x, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)
        return self.final_norm(out)  # (B, T', D)


# ----------------------------------------------------------------------
# Heads
# ----------------------------------------------------------------------
class ReconstructionHead(nn.Module):
    """Predict (numeric values, categorical logits) per position.

    Numeric head outputs F real numbers; we compare via MSE against the
    z-scored ground truth (masked positions only).
    Categorical heads output V_i logits per column; we compare via CE.
    """

    def __init__(self, tokenizer: TokenizerArtifact, d_model: int) -> None:
        super().__init__()
        self.numeric_head = nn.Linear(d_model, tokenizer.n_numeric)
        self.cat_heads = nn.ModuleList(
            [
                nn.Linear(d_model, tokenizer.cat_vocab_size(col))
                for col in tokenizer.categorical_cols
            ]
        )

    def forward(self, hidden: Tensor) -> tuple[Tensor, list[Tensor]]:
        """hidden: (B, T', D). Returns (num_pred (B, T', F), cat_logits list of (B, T', V_i))."""
        num_pred = self.numeric_head(hidden)
        cat_logits = [head(hidden) for head in self.cat_heads]
        return num_pred, cat_logits


class ContrastiveHead(nn.Module):
    """SimCLR-style projection head: (B, D) -> (B, proj_dim) L2-normalized."""

    def __init__(self, d_model: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, cls_hidden: Tensor) -> Tensor:
        z = self.net(cls_hidden)  # (B, proj_dim)
        return torch.nn.functional.normalize(z, dim=-1)


class ClassificationHead(nn.Module):
    """Binary classifier on top of mean-pooled or CLS sequence embedding."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, D). Returns logits (B,)."""
        return self.net(x).squeeze(-1)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
