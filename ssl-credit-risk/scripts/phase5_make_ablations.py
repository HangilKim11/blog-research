"""Build the Phase 5 ablation feature parquets.

Pipeline:
1. Train ONE LightGBM fold on the Phase-1 hand-only features and dump the
   full feature-importance ranking (by gain).
2. Identify the top-K hand features to drop.
3. Write three new feature parquets used by the ablation study:
   - features_hand_minus_topK.parquet           (1,291 - K cols)
   - features_hand_minus_topK_plus_ssl.parquet  ((1,291 - K) + 128 cols)
   - features_ssl_only.parquet                  (128 cols + target)

Reads:
- data/processed/v1/features_lgbm.parquet
- data/processed/v1/features_ssl_hybrid.parquet
- data/splits/v1.parquet
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

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=False)

KEY_COL = "customer_ID"
TARGET_COL = "target"
SPLIT_COL = "split"
FOLD_COL = "fold"
TEST_SPLIT = "test"

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
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


def _fold0_importance(base_path: Path, splits_path: Path) -> tuple[list[str], list[float]]:
    """Train fold-0 LGB on hand-only features; return (feature_cols, gain_importance)."""
    feats = pl.read_parquet(base_path)
    splits = pl.read_parquet(splits_path).select([KEY_COL, SPLIT_COL, FOLD_COL, TARGET_COL])
    if TARGET_COL in feats.columns:
        feats = feats.drop(TARGET_COL)
    joined = feats.join(splits, on=KEY_COL)

    trainval = joined.filter(pl.col(SPLIT_COL) != TEST_SPLIT)
    feature_cols: list[str] = [c for c in feats.columns if c != KEY_COL]

    X = trainval.select(feature_cols).to_numpy().astype(np.float32)
    y = trainval[TARGET_COL].to_numpy().astype(np.float64)
    fold = trainval[FOLD_COL].to_numpy().astype(np.int64)
    val_mask = fold == 0
    console.print(
        f"[bold]fitting hand-only fold-0[/]  features={len(feature_cols)}  "
        f"train={(~val_mask).sum():,}  val={val_mask.sum():,}"
    )

    train_ds = lgb.Dataset(X[~val_mask], label=y[~val_mask])
    val_ds = lgb.Dataset(X[val_mask], label=y[val_mask], reference=train_ds)
    booster = lgb.train(
        LGB_PARAMS,
        train_ds,
        num_boost_round=2000,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
    )
    gains = booster.feature_importance(importance_type="gain").tolist()
    return feature_cols, [float(g) for g in gains]


@app.command()
def main(
    base_path: Path = typer.Option(Path("data/processed/v1/features_lgbm.parquet"), "--base"),
    ssl_path: Path = typer.Option(Path("data/processed/v1/features_ssl_hybrid.parquet"), "--ssl"),
    splits_path: Path = typer.Option(Path("data/splits/v1.parquet"), "--splits"),
    k: int = typer.Option(100, "--k", help="how many top hand features to ablate."),
    out_dir: Path = typer.Option(Path("data/processed/v1"), "--out-dir"),
    importance_out: Path = typer.Option(
        Path("reports/feature_importance_hand_only.json"), "--imp-out"
    ),
) -> None:
    """Ablation prep: importance dump + 3 subset parquets."""
    cols, gains = _fold0_importance(base_path, splits_path)
    pairs = sorted(zip(cols, gains, strict=True), key=lambda p: -p[1])

    top_k_cols = [c for c, _ in pairs[:k]]
    importance_out.parent.mkdir(parents=True, exist_ok=True)
    importance_out.write_text(
        json.dumps(
            {
                "k": k,
                "top_k_columns": top_k_cols,
                "all_features_by_gain": [{"feature": c, "gain": g} for c, g in pairs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"[bold green]wrote[/] {importance_out}")

    # ---- subset 1: hand - top-K ----
    base = pl.read_parquet(base_path)
    keep_cols = [KEY_COL] + [c for c in cols if c not in set(top_k_cols)]
    if TARGET_COL in base.columns:
        keep_cols.append(TARGET_COL)
    sub_minus = base.select(keep_cols)
    out_minus = out_dir / f"features_hand_minus_top{k}.parquet"
    sub_minus.write_parquet(out_minus, compression="zstd")

    # ---- subset 2: (hand - top-K) + SSL ----
    ssl = pl.read_parquet(ssl_path)
    sub_minus_plus = sub_minus.join(ssl, on=KEY_COL, how="left")
    out_minus_plus = out_dir / f"features_hand_minus_top{k}_plus_ssl.parquet"
    sub_minus_plus.write_parquet(out_minus_plus, compression="zstd")

    # ---- subset 3: SSL only (+ customer_ID + target from base) ----
    target_col = (
        base.select([KEY_COL, TARGET_COL]) if TARGET_COL in base.columns else base.select(KEY_COL)
    )
    sub_ssl_only = target_col.join(ssl, on=KEY_COL, how="left")
    out_ssl_only = out_dir / "features_ssl_only.parquet"
    sub_ssl_only.write_parquet(out_ssl_only, compression="zstd")

    table = Table(title="Phase 5 ablation subsets", show_lines=False)
    table.add_column("subset", style="cyan")
    table.add_column("rows", justify="right")
    table.add_column("cols", justify="right")
    table.add_column("path")
    table.add_row(
        f"(ii) hand - top{k}", f"{sub_minus.height:,}", str(len(sub_minus.columns)), str(out_minus)
    )
    table.add_row(
        f"(iii) (hand - top{k}) + SSL",
        f"{sub_minus_plus.height:,}",
        str(len(sub_minus_plus.columns)),
        str(out_minus_plus),
    )
    table.add_row(
        "(iv) SSL only",
        f"{sub_ssl_only.height:,}",
        str(len(sub_ssl_only.columns)),
        str(out_ssl_only),
    )
    console.print(table)
    console.print("(i) full hand is already at data/processed/v1/features_lgbm.parquet")
    console.print("(v) full hand + SSL is already at data/processed/v1/features_augmented.parquet")


if __name__ == "__main__":
    app()
