"""Re-fit ONE fold on the augmented features and rank feature importance.

Quick way to answer: did LightGBM actually use the SSL embedding columns?
We train a single fold-0 model (so we don't pay for the full 5-fold rerun),
then dump:
  - top-K most important features (any type)
  - SSL embedding columns sorted by importance
  - aggregate share-of-importance for hand-crafted vs SSL columns
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table
from sklearn.metrics import roc_auc_score

from amex.evaluation.metrics import amex_metric_components

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=False)

KEY_COL = "customer_ID"
TARGET_COL = "target"
SPLIT_COL = "split"
FOLD_COL = "fold"
TEST_SPLIT = "test"


@app.command()
def main(  # noqa: PLR0915 -- linear one-shot script; splitting hurts readability
    features_path: Path = typer.Option(
        Path("data/processed/v1/features_augmented.parquet"), "--features"
    ),
    splits_path: Path = typer.Option(Path("data/splits/v1.parquet"), "--splits"),
    out_path: Path = typer.Option(Path("reports/feature_importance_augmented.json"), "--out"),
    fold_id: int = typer.Option(0, "--fold"),
    n_estimators: int = typer.Option(2000, "--n-estimators"),
    ssl_prefix: str = typer.Option("ssl_hybrid_emb_", "--ssl-prefix"),
) -> None:
    """One-fold importance dump, no W&B, no CV."""
    feats = pl.read_parquet(features_path)
    splits = pl.read_parquet(splits_path).select([KEY_COL, SPLIT_COL, FOLD_COL, TARGET_COL])
    if TARGET_COL in feats.columns:
        feats = feats.drop(TARGET_COL)
    joined = feats.join(splits, on=KEY_COL)

    trainval = joined.filter(pl.col(SPLIT_COL) != TEST_SPLIT)
    feature_cols: list[str] = [c for c in feats.columns if c != KEY_COL]

    X = trainval.select(feature_cols).to_numpy().astype(np.float32)
    y = trainval[TARGET_COL].to_numpy().astype(np.float64)
    fold = trainval[FOLD_COL].to_numpy().astype(np.int64)
    val_mask = fold == fold_id

    console.print(
        f"[bold]features[/]   total={len(feature_cols)}  "
        f"ssl={sum(1 for c in feature_cols if c.startswith(ssl_prefix))}"
    )
    console.print(f"[bold]fold {fold_id}[/]  train={(~val_mask).sum()}  val={val_mask.sum()}")

    train_ds = lgb.Dataset(X[~val_mask], label=y[~val_mask])
    val_ds = lgb.Dataset(X[val_mask], label=y[val_mask], reference=train_ds)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,  # bumped to converge faster for a one-shot
        "num_leaves": 100,
        "min_child_samples": 2400,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "colsample_bytree": 0.4,
        "subsample": 0.8,
        "subsample_freq": 5,
        "max_depth": -1,
        "max_bin": 255,
        "n_jobs": -1,
        "verbosity": -1,
        "random_state": 42,
    }
    booster = lgb.train(
        params,
        train_ds,
        num_boost_round=n_estimators,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )

    val_pred = booster.predict(X[val_mask], num_iteration=booster.best_iteration)
    m, g, d = amex_metric_components(y[val_mask], val_pred)
    auc = float(roc_auc_score(y[val_mask], val_pred))
    console.print(f"[bold]fold-{fold_id} val[/] amex={m:.5f} auc={auc:.5f}")

    # Importance: split count (default) + gain
    imp_gain = booster.feature_importance(importance_type="gain")
    imp_split = booster.feature_importance(importance_type="split")
    total_gain = float(imp_gain.sum())
    total_split = int(imp_split.sum())

    pairs = list(zip(feature_cols, imp_gain.tolist(), imp_split.tolist(), strict=True))
    is_ssl = lambda c: c.startswith(ssl_prefix)  # noqa: E731
    ssl_pairs = [p for p in pairs if is_ssl(p[0])]
    hand_pairs = [p for p in pairs if not is_ssl(p[0])]

    ssl_gain = sum(p[1] for p in ssl_pairs)
    hand_gain = sum(p[1] for p in hand_pairs)
    ssl_split = sum(p[2] for p in ssl_pairs)
    hand_split = sum(p[2] for p in hand_pairs)

    # Top-N by gain (any type)
    pairs_sorted = sorted(pairs, key=lambda p: -p[1])
    top_n = 25
    table = Table(title=f"Top {top_n} features by gain (any type)", show_lines=False)
    table.add_column("rank", justify="right")
    table.add_column("feature")
    table.add_column("gain", justify="right")
    table.add_column("split count", justify="right")
    table.add_column("type", style="cyan")
    for i, (c, gv, sv) in enumerate(pairs_sorted[:top_n], 1):
        table.add_row(str(i), c, f"{gv:,.0f}", f"{sv:,}", "SSL" if is_ssl(c) else "hand")
    console.print(table)

    # Top SSL features specifically
    ssl_sorted = sorted(ssl_pairs, key=lambda p: -p[1])
    table2 = Table(title="Top-10 SSL embedding features by gain", show_lines=False)
    table2.add_column("rank", justify="right")
    table2.add_column("feature")
    table2.add_column("gain", justify="right")
    table2.add_column("split count", justify="right")
    table2.add_column("rank in all", justify="right")
    rank_lookup = {c: i + 1 for i, (c, _, _) in enumerate(pairs_sorted)}
    for i, (c, gv, sv) in enumerate(ssl_sorted[:10], 1):
        table2.add_row(str(i), c, f"{gv:,.0f}", f"{sv:,}", str(rank_lookup[c]))
    console.print(table2)

    # Aggregate share-of-importance
    table3 = Table(title="Importance share: hand-crafted vs SSL", show_lines=False)
    table3.add_column("group", style="cyan")
    table3.add_column("n cols", justify="right")
    table3.add_column("gain", justify="right")
    table3.add_column("gain %", justify="right")
    table3.add_column("split count", justify="right")
    table3.add_column("split %", justify="right")
    table3.add_row(
        "hand-crafted",
        f"{len(hand_pairs)}",
        f"{hand_gain:,.0f}",
        f"{100 * hand_gain / total_gain:.2f}",
        f"{hand_split:,}",
        f"{100 * hand_split / total_split:.2f}",
    )
    table3.add_row(
        f"SSL ({ssl_prefix}*)",
        f"{len(ssl_pairs)}",
        f"{ssl_gain:,.0f}",
        f"{100 * ssl_gain / total_gain:.2f}",
        f"{ssl_split:,}",
        f"{100 * ssl_split / total_split:.2f}",
    )
    console.print(table3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fold_id": fold_id,
        "n_features_total": len(feature_cols),
        "n_features_ssl": len(ssl_pairs),
        "n_features_hand": len(hand_pairs),
        "val_amex": m,
        "val_g": g,
        "val_d": d,
        "val_auc": auc,
        "best_iter": int(booster.best_iteration or 0),
        "importance_share": {
            "hand_gain_pct": 100 * hand_gain / total_gain,
            "ssl_gain_pct": 100 * ssl_gain / total_gain,
            "hand_split_pct": 100 * hand_split / total_split,
            "ssl_split_pct": 100 * ssl_split / total_split,
        },
        "top_25_by_gain": [
            {"feature": c, "gain": gv, "split_count": sv, "type": "SSL" if is_ssl(c) else "hand"}
            for c, gv, sv in pairs_sorted[:25]
        ],
        "top_10_ssl_by_gain": [
            {"feature": c, "gain": gv, "split_count": sv, "rank_in_all": rank_lookup[c]}
            for c, gv, sv in ssl_sorted[:10]
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    console.print(f"[bold green]wrote[/] {out_path}")


if __name__ == "__main__":
    app()
