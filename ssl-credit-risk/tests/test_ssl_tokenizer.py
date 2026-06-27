"""Tokenizer + dataset round-trip tests on a small fixture parquet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from amex.data.sequence_builder import DATE_COL, KEY_COL
from amex.ssl.dataset import AmexSSLDataset, collate_ssl_samples
from amex.ssl.tokenizer import (
    MAX_SEQ_LEN,
    MISSING_CAT_CODE,
    TokenizerArtifact,
    encode_customer,
    fit_tokenizer,
)


def _make_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny train partition + splits parquet for tests."""
    customers = [f"cust_{i:03d}" for i in range(20)]
    rows = []
    for ci, c in enumerate(customers):
        n_stmts = 5 + (ci % 6)  # 5..10 statements
        for t in range(n_stmts):
            rows.append(
                {
                    KEY_COL: c,
                    DATE_COL: f"2022-{1 + t:02d}-01",
                    "P_2": 0.5 + 0.1 * t + 0.01 * ci,  # numeric
                    "B_1": float("nan") if t == 0 else 1.0 * ci,  # numeric with NaN
                    "B_30": ci % 3,  # categorical (small int)
                    "D_63": "CR" if t % 2 == 0 else "CO",  # categorical (string)
                }
            )
    frame = pl.DataFrame(rows).with_columns(pl.col(DATE_COL).str.to_date())
    part_dir = tmp_path / "train" / "_partition=00"
    part_dir.mkdir(parents=True)
    pq_path = part_dir / "part.parquet"
    frame.write_parquet(pq_path)

    # splits: first 16 customers train, next 2 val, last 2 test
    splits = pl.DataFrame(
        {
            KEY_COL: customers,
            "target": [int(i % 4 == 0) for i in range(20)],
            "split": ["train"] * 16 + ["val"] * 2 + ["test"] * 2,
            "fold": [i % 5 for i in range(16)] + [-1] * 4,
        }
    )
    splits_path = tmp_path / "splits.parquet"
    splits.write_parquet(splits_path)

    return pq_path, splits_path


def test_fit_tokenizer_train_only(tmp_path: Path) -> None:
    pq_path, splits_path = _make_fixture(tmp_path)
    glob = str(pq_path.parent.parent / "**" / "*.parquet")
    tok = fit_tokenizer(glob, splits_path)

    # 2 numeric, 2 categorical
    assert set(tok.numeric_cols) == {"P_2", "B_1"}
    assert set(tok.categorical_cols) == {"B_30", "D_63"}

    # vocab: B_30 has values 0,1,2 -> codes 1,2,3; D_63 has CR,CO -> codes 1,2
    assert tok.cat_vocab_size("B_30") == 4  # 3 values + MISSING
    assert tok.cat_vocab_size("D_63") == 3  # 2 values + MISSING

    # numeric stats are non-trivial
    assert tok.numeric_std["P_2"] > 0
    assert 0.0 <= tok.numeric_mean["P_2"] <= 2.0


def test_tokenizer_save_load(tmp_path: Path) -> None:
    pq_path, splits_path = _make_fixture(tmp_path)
    glob = str(pq_path.parent.parent / "**" / "*.parquet")
    tok = fit_tokenizer(glob, splits_path)

    json_path = tmp_path / "tokenizer.json"
    tok.save(json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["numeric_cols"]
    tok2 = TokenizerArtifact.load(json_path)
    assert tok2.numeric_cols == tok.numeric_cols
    assert tok2.numeric_mean == tok.numeric_mean
    assert tok2.categorical_vocab == tok.categorical_vocab


def test_encode_customer_shapes(tmp_path: Path) -> None:
    pq_path, splits_path = _make_fixture(tmp_path)
    glob = str(pq_path.parent.parent / "**" / "*.parquet")
    tok = fit_tokenizer(glob, splits_path)

    raw = pl.read_parquet(pq_path).filter(pl.col(KEY_COL) == "cust_005").sort(DATE_COL)
    sample = encode_customer(tok, raw, max_len=MAX_SEQ_LEN)

    assert sample["numeric_values"].shape == (MAX_SEQ_LEN, tok.n_numeric)
    assert sample["numeric_mask"].shape == (MAX_SEQ_LEN, tok.n_numeric)
    assert sample["categorical_ids"].shape == (MAX_SEQ_LEN, tok.n_categorical)
    assert sample["attention_mask"].shape == (MAX_SEQ_LEN,)
    assert sample["seq_len"] == raw.height
    assert sample["customer_id"] == "cust_005"

    # padding rows are False in attention_mask
    assert sample["attention_mask"][sample["seq_len"] :].sum() == 0
    # padding rows have numeric_mask == True (counted as missing)
    assert sample["numeric_mask"][sample["seq_len"] :].all()


def test_encode_handles_missing_categorical_value(tmp_path: Path) -> None:
    pq_path, splits_path = _make_fixture(tmp_path)
    glob = str(pq_path.parent.parent / "**" / "*.parquet")
    tok = fit_tokenizer(glob, splits_path)

    raw = pl.DataFrame(
        {
            KEY_COL: ["cust_999"],
            DATE_COL: ["2022-01-01"],
            "P_2": [0.5],
            "B_1": [1.0],
            "B_30": [None],  # missing
            "D_63": ["XX"],  # unknown value, never seen
        }
    ).with_columns(pl.col(DATE_COL).str.to_date())

    sample = encode_customer(tok, raw)
    # B_30 col index
    b30_idx = tok.categorical_cols.index("B_30")
    d63_idx = tok.categorical_cols.index("D_63")
    assert int(sample["categorical_ids"][0, b30_idx]) == MISSING_CAT_CODE
    assert int(sample["categorical_ids"][0, d63_idx]) == MISSING_CAT_CODE


def test_numeric_zscore_zero_mean_unit_std(tmp_path: Path) -> None:
    """Across many encoded samples from train customers, normalized numerics
    should be approximately mean=0 std=1."""
    pq_path, splits_path = _make_fixture(tmp_path)
    glob = str(pq_path.parent.parent / "**" / "*.parquet")
    tok = fit_tokenizer(glob, splits_path)

    frame = pl.read_parquet(pq_path).sort([KEY_COL, DATE_COL])
    encoded = []
    for cust_id in frame[KEY_COL].unique().to_list()[:16]:  # train customers only
        sub = frame.filter(pl.col(KEY_COL) == cust_id).sort(DATE_COL)
        s = encode_customer(tok, sub)
        # only count valid rows
        m = s["attention_mask"]
        encoded.append(s["numeric_values"][m].numpy())
    vals = np.concatenate(encoded, axis=0)
    # ignore P_2 (column 0) only — average across all positions
    col_mean = vals.mean(axis=0)
    col_std = vals.std(axis=0)
    # P_2 was z-scored on train so its mean ≈ 0 std ≈ 1 (within tolerance from sample noise)
    assert abs(col_mean[tok.numeric_cols.index("P_2")]) < 0.5
    assert 0.5 < col_std[tok.numeric_cols.index("P_2")] < 1.6


def test_dataset_yields_correct_shapes(tmp_path: Path) -> None:
    pq_path, splits_path = _make_fixture(tmp_path)
    glob = str(pq_path.parent.parent / "**" / "*.parquet")
    tok = fit_tokenizer(glob, splits_path)

    splits = pl.read_parquet(splits_path)
    train_ids = set(splits.filter(pl.col("split") == "train")[KEY_COL].to_list())

    ds = AmexSSLDataset(
        partition_glob=glob,
        tokenizer=tok,
        customer_id_filter=train_ids,
        shuffle_partitions=False,
        shuffle_within_partition=False,
        seed=42,
    )
    samples = list(ds)
    assert len(samples) == len(train_ids)

    batch = collate_ssl_samples(samples[:4])
    assert batch["numeric_values"].shape == (4, MAX_SEQ_LEN, tok.n_numeric)
    assert batch["categorical_ids"].shape == (4, MAX_SEQ_LEN, tok.n_categorical)
    assert batch["attention_mask"].shape == (4, MAX_SEQ_LEN)
    assert len(batch["customer_id"]) == 4


def test_dataset_filter_excludes_unwanted_customers(tmp_path: Path) -> None:
    pq_path, splits_path = _make_fixture(tmp_path)
    glob = str(pq_path.parent.parent / "**" / "*.parquet")
    tok = fit_tokenizer(glob, splits_path)

    test_ids = {"cust_018", "cust_019"}
    ds = AmexSSLDataset(
        partition_glob=glob,
        tokenizer=tok,
        customer_id_filter=test_ids,
        shuffle_partitions=False,
        shuffle_within_partition=False,
    )
    samples = list(ds)
    assert {s["customer_id"] for s in samples} == test_ids


@pytest.mark.parametrize("max_len", [3, 7, 13])
def test_truncate_keeps_recent_statements(max_len: int, tmp_path: Path) -> None:
    """When seq > max_len, the LAST max_len statements should be kept."""
    pq_path, splits_path = _make_fixture(tmp_path)
    glob = str(pq_path.parent.parent / "**" / "*.parquet")
    tok = fit_tokenizer(glob, splits_path)

    # cust_005 has 10 statements (>max_len=3)
    raw = pl.read_parquet(pq_path).filter(pl.col(KEY_COL) == "cust_005").sort(DATE_COL)
    sample = encode_customer(tok, raw, max_len=max_len)
    assert sample["seq_len"] == min(raw.height, max_len)
    # the surviving rows should be the last ones (highest dates)
    if raw.height > max_len:
        kept_p2 = sample["numeric_values"][:max_len, tok.numeric_cols.index("P_2")]
        last_p2_normalized = (
            raw[tok.numeric_cols].tail(max_len).to_numpy()[:, tok.numeric_cols.index("P_2")]
            - tok.numeric_mean["P_2"]
        ) / tok.numeric_std["P_2"]
        torch.testing.assert_close(
            kept_p2, torch.from_numpy(last_p2_normalized.astype("float32")), atol=1e-5, rtol=1e-5
        )
