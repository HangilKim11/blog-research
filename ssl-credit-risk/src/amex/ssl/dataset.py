"""Streaming PyTorch dataset that turns partitioned Parquet into SSL samples.

Each yielded sample is one customer's tensors (output of
``encode_customer``). The default collate stacks B samples into batches
of shape ``(B, max_len, F)`` -- max_len is fixed at the tokenizer
default (13) so no extra padding is needed.

Usage
-----
    tok = TokenizerArtifact.load(Path("data/processed/v1/tokenizer.json"))
    ds = AmexSSLDataset(
        partition_glob="data/processed/v1/train/**/*.parquet",
        tokenizer=tok,
        customer_id_filter=set(train_ids),
    )
    loader = DataLoader(ds, batch_size=256, num_workers=2)
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from glob import glob as _stdglob  # Path.glob can't handle absolute patterns
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import IterableDataset, get_worker_info

from amex.data.sequence_builder import DATE_COL, KEY_COL
from amex.ssl.tokenizer import MAX_SEQ_LEN, TokenizerArtifact, encode_customer


class AmexSSLDataset(IterableDataset[dict[str, Any]]):
    """Yields one encoded customer sample at a time, partition by partition.

    Worker-aware: ``DataLoader(num_workers=k)`` causes each worker to scan a
    disjoint subset of partition files. Within a partition, shuffling of
    customer order can be enabled to avoid sorted-batch artifacts.
    """

    def __init__(
        self,
        partition_glob: str,
        tokenizer: TokenizerArtifact,
        *,
        customer_id_filter: set[str] | None = None,
        max_len: int = MAX_SEQ_LEN,
        shuffle_partitions: bool = True,
        shuffle_within_partition: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        # stdlib glob handles both relative and absolute patterns with '**'.
        self.partition_files = [
            Path(p)
            for p in sorted(_stdglob(partition_glob, recursive=True))  # noqa: PTH207
        ]
        if not self.partition_files:
            msg = f"no partition files matched glob: {partition_glob}"
            raise FileNotFoundError(msg)
        self.tokenizer = tokenizer
        self.customer_id_filter = customer_id_filter
        self.max_len = max_len
        self.shuffle_partitions = shuffle_partitions
        self.shuffle_within_partition = shuffle_within_partition
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        files = list(self.partition_files)
        rng = random.Random(self.seed + (worker.id if worker else 0))
        if self.shuffle_partitions:
            rng.shuffle(files)
        if worker is not None:
            files = files[worker.id :: worker.num_workers]

        filter_list = list(self.customer_id_filter) if self.customer_id_filter else None
        for path in files:
            yield from self._iter_partition(path, filter_list, rng)

    def _iter_partition(
        self, path: Path, filter_list: list[str] | None, rng: random.Random
    ) -> Iterator[dict[str, Any]]:
        frame = pl.read_parquet(path)
        if filter_list is not None:
            frame = frame.filter(pl.col(KEY_COL).is_in(filter_list))
        if frame.height == 0:
            return
        frame = frame.sort([KEY_COL, DATE_COL])

        # Materialize per-customer slices via partition_by (faster than group_by
        # iteration for many small groups).
        groups: list[pl.DataFrame] = frame.partition_by(KEY_COL, maintain_order=True)
        if self.shuffle_within_partition:
            rng.shuffle(groups)
        for g in groups:
            yield encode_customer(self.tokenizer, g, max_len=self.max_len)


def collate_ssl_samples(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[str]]:
    """Default collate: stack tensors, keep customer_ids as a list."""
    out: dict[str, torch.Tensor | list[str]] = {
        "numeric_values": torch.stack([s["numeric_values"] for s in samples]),
        "numeric_mask": torch.stack([s["numeric_mask"] for s in samples]),
        "categorical_ids": torch.stack([s["categorical_ids"] for s in samples]),
        "attention_mask": torch.stack([s["attention_mask"] for s in samples]),
        "seq_len": torch.tensor([s["seq_len"] for s in samples], dtype=torch.int64),
        "customer_id": [s["customer_id"] for s in samples],
    }
    return out
