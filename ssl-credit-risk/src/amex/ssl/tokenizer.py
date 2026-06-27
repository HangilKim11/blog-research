"""Customer-sequence tokenizer for SSL pretraining.

The pretraining pipeline needs to turn variable-length customer statement
sequences into fixed-shape tensors. This module:

1. **Fits** per-feature statistics on the *train-split* only:
   - numeric columns -> mean, std (z-score normalization)
   - categorical columns -> integer vocab (code 0 reserved for MISSING)
2. **Encodes** a customer's polars rows into a padded sample dict of tensors.
3. **Saves / loads** the fitted artifact as JSON so the same encoding can be
   reused by linear-probe and fine-tune downstream.

Conventions
-----------
- Numeric NaN -> value=0.0, mask=True. The encoder is expected to read both.
- Categorical NaN / unknown -> code 0 (MISSING).
- Padding -> value=0.0, mask=True, attention_mask=False.
- Max sequence length is hard-capped at 13 (AMEX data has at most 13 monthly
  statements per customer).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from amex.data.sequence_builder import CATEGORICAL_COLS, DATE_COL, KEY_COL

MAX_SEQ_LEN = 13
MISSING_CAT_CODE = 0  # reserved for NaN / unknown
RESERVED_COLS = (KEY_COL, DATE_COL, "_partition", "target")


@dataclass
class TokenizerArtifact:
    """Fitted feature-encoding state. JSON-serializable."""

    numeric_cols: list[str]
    categorical_cols: list[str]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    # vocab[col_name] = mapping {raw_value (as str): int_code}.
    # Raw values are stringified for JSON portability; code 0 is reserved.
    categorical_vocab: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def n_numeric(self) -> int:
        return len(self.numeric_cols)

    @property
    def n_categorical(self) -> int:
        return len(self.categorical_cols)

    def cat_vocab_size(self, col: str) -> int:
        """Embedding table size for column ``col`` (includes MISSING)."""
        return len(self.categorical_vocab[col]) + 1  # +1 for MISSING

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TokenizerArtifact:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)


def _split_columns(all_cols: list[str]) -> tuple[list[str], list[str]]:
    """Partition raw schema into (numeric, categorical) feature lists."""
    cat_set = set(CATEGORICAL_COLS)
    numeric = [c for c in all_cols if c not in cat_set and c not in RESERVED_COLS]
    categorical = [c for c in all_cols if c in cat_set]
    return numeric, categorical


def fit_tokenizer(
    train_partition_glob: str,
    splits_path: Path,
    *,
    splits_to_use: tuple[str, ...] = ("train",),
) -> TokenizerArtifact:
    """Fit per-feature stats on the listed splits only (default: train).

    Parameters
    ----------
    train_partition_glob : str
        glob matching the hash-partitioned train parquet files, e.g.
        ``"data/processed/v1/train/**/*.parquet"``.
    splits_path : Path
        path to ``data/splits/v1.parquet``.
    splits_to_use : tuple of str
        which split labels to include in the fit. Default ``("train",)``
        avoids any leakage from val/test customers.
    """
    splits = pl.read_parquet(splits_path).select([KEY_COL, "split"])
    use_ids = splits.filter(pl.col("split").is_in(list(splits_to_use))).select(KEY_COL)

    lf = pl.scan_parquet(train_partition_glob).join(use_ids.lazy(), on=KEY_COL, how="inner")

    schema_cols = lf.collect_schema().names()
    numeric, categorical = _split_columns(schema_cols)

    # --- numeric stats: one pass for mean+std across all cols ---
    stat_exprs = [pl.col(c).mean().alias(f"{c}__mean") for c in numeric] + [
        pl.col(c).std().alias(f"{c}__std") for c in numeric
    ]
    stats_row = lf.select(stat_exprs).collect().to_dicts()[0]

    def _mean(value: Any) -> float:
        """Real mean or 0.0 for an all-NaN column."""
        if value is None:
            return 0.0
        f = float(value)
        return 0.0 if np.isnan(f) else f

    def _std(value: Any) -> float:
        """Real std or 1.0 for a constant / all-NaN column (avoids div-by-zero)."""
        if value is None:
            return 1.0
        f = float(value)
        if np.isnan(f) or f == 0.0:
            return 1.0
        return f

    numeric_mean = {c: _mean(stats_row[f"{c}__mean"]) for c in numeric}
    numeric_std = {c: _std(stats_row[f"{c}__std"]) for c in numeric}

    # --- categorical vocab: one column at a time so we get a clean list ---
    vocab: dict[str, dict[str, int]] = {}
    for col in categorical:
        uniques = lf.select(col).drop_nulls().unique().sort(col).collect().to_series().to_list()
        vocab[col] = {str(v): i + 1 for i, v in enumerate(uniques)}  # 0 = MISSING

    return TokenizerArtifact(
        numeric_cols=numeric,
        categorical_cols=categorical,
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        categorical_vocab=vocab,
    )


def encode_customer(
    tok: TokenizerArtifact,
    customer_df: pl.DataFrame,
    *,
    max_len: int = MAX_SEQ_LEN,
) -> dict[str, Any]:
    """Encode one customer's rows (already sorted by date) into padded tensors.

    Returns a dict with shapes:
        numeric_values  (max_len, F_num) float32
        numeric_mask    (max_len, F_num) bool   -- True where original was NaN OR pad
        categorical_ids (max_len, F_cat) int64  -- code 0 = MISSING/pad
        attention_mask  (max_len,) bool         -- True for valid (non-pad) timesteps
        seq_len         int (actual rows, <= max_len)
        customer_id     str
    """
    # Clip to last `max_len` rows (most-recent statements take priority).
    if customer_df.height > max_len:
        customer_df = customer_df.tail(max_len)
    seq_len = customer_df.height

    # --- numeric ---
    F_num = tok.n_numeric
    num_vals = np.zeros((max_len, F_num), dtype=np.float32)
    num_mask = np.ones((max_len, F_num), dtype=bool)  # True = missing or pad
    if seq_len > 0 and F_num > 0:
        raw = customer_df.select(tok.numeric_cols).to_numpy().astype(np.float32)
        is_nan = np.isnan(raw)
        raw = np.where(is_nan, 0.0, raw)
        # z-score
        mean = np.array([tok.numeric_mean[c] for c in tok.numeric_cols], dtype=np.float32)
        std = np.array([tok.numeric_std[c] for c in tok.numeric_cols], dtype=np.float32)
        raw = (raw - mean[None, :]) / std[None, :]
        # zero out NaNs after normalization
        raw = np.where(is_nan, 0.0, raw)
        num_vals[:seq_len] = raw
        num_mask[:seq_len] = is_nan

    # --- categorical ---
    F_cat = tok.n_categorical
    cat_ids = np.zeros((max_len, F_cat), dtype=np.int64)
    if seq_len > 0 and F_cat > 0:
        for j, col in enumerate(tok.categorical_cols):
            vocab = tok.categorical_vocab[col]
            series = customer_df[col].to_list()
            for i, v in enumerate(series):
                if v is None:
                    cat_ids[i, j] = MISSING_CAT_CODE
                else:
                    cat_ids[i, j] = vocab.get(str(v), MISSING_CAT_CODE)

    attention_mask = np.zeros(max_len, dtype=bool)
    attention_mask[:seq_len] = True

    return {
        "customer_id": customer_df[KEY_COL][0],
        "numeric_values": torch.from_numpy(num_vals),
        "numeric_mask": torch.from_numpy(num_mask),
        "categorical_ids": torch.from_numpy(cat_ids),
        "attention_mask": torch.from_numpy(attention_mask),
        "seq_len": int(seq_len),
    }
