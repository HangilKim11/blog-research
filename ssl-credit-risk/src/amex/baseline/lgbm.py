"""LightGBM baseline trained on the customer-level engineered features.

Trains 5-fold customer-level CV using the canonical splits in
``data/splits/v1.parquet`` (fold column ranges 0..4 for trainval; test
holdout has fold = -1 and is scored once at the end with the average of
the 5 fold models).

Logs every fold to Weights & Biases (project ``amex-ssl``, tag
``baseline``) and saves OOF predictions to ``data/processed/v1/oof_lgbm.parquet``.

Reports per fold and aggregated:
- AMEX competition metric (M)
- AUC
- Kolmogorov-Smirnov (KS) statistic
- log-loss

Run
---
    uv run python -m amex.baseline.lgbm                  # full run, W&B online
    uv run python -m amex.baseline.lgbm --no-wandb       # disable W&B
    uv run python -m amex.baseline.lgbm --quick          # 1 fold, 500 trees -- smoke
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightgbm as lgb
import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table
from sklearn.metrics import log_loss, roc_auc_score

from amex.evaluation.metrics import amex_metric_components

if TYPE_CHECKING:
    from numpy.typing import NDArray

KEY_COL = "customer_ID"
TARGET_COL = "target"
SPLIT_COL = "split"
FOLD_COL = "fold"
TEST_SPLIT = "test"
TEST_FOLD_SENTINEL = -1
WANDB_PROJECT = "amex-ssl"

FloatArray = "NDArray[np.float64]"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


# AMEX-grade LightGBM config (close to the public 1st-place writeup):
# slow learning rate, generous num_leaves, strong regularization. Tuned in
# Session 2 -- here we just want a defensible baseline.
LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.01,
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
N_ESTIMATORS = 10_500
EARLY_STOP = 200


def _as_f64(arr: Any) -> NDArray[np.float64]:
    """Coerce any array-like to a contiguous 1-D float64 numpy array."""
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float64).reshape(-1))


def _ks(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Kolmogorov-Smirnov statistic between positive and negative score distributions."""
    pos = np.sort(y_pred[y_true == 1])
    neg = np.sort(y_pred[y_true == 0])
    if pos.size == 0 or neg.size == 0:
        return 0.0
    grid = np.concatenate([pos, neg])
    cdf_pos = np.searchsorted(pos, grid, side="right") / pos.size
    cdf_neg = np.searchsorted(neg, grid, side="right") / neg.size
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def _all_metrics(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> dict[str, float]:
    """AMEX + AUC + KS + log-loss in one shot."""
    m, g, d = amex_metric_components(y_true, y_pred)
    return {
        "amex": m,
        "amex_g": g,
        "amex_d": d,
        "auc": float(roc_auc_score(y_true, y_pred)),
        "ks": _ks(y_true, y_pred),
        "log_loss": float(log_loss(y_true, np.clip(y_pred, 1e-15, 1 - 1e-15))),
    }


def _lgb_amex_eval(y_pred: NDArray[np.float64], dataset: lgb.Dataset) -> tuple[str, float, bool]:
    """LightGBM custom-eval callback returning the AMEX metric."""
    y_true = _as_f64(dataset.get_label())
    m, _, _ = amex_metric_components(y_true, _as_f64(y_pred))
    return "amex", m, True  # higher is better


def _load_features_and_splits(
    features_path: Path, splits_path: Path
) -> tuple[pl.DataFrame, list[str]]:
    """Join features <- splits on customer_ID and report sizes."""
    feats = pl.read_parquet(features_path)
    splits = pl.read_parquet(splits_path).select([KEY_COL, SPLIT_COL, FOLD_COL, TARGET_COL])
    if TARGET_COL in feats.columns:
        feats = feats.drop(TARGET_COL)
    joined = feats.join(splits, on=KEY_COL)

    feature_cols = [c for c in feats.columns if c != KEY_COL]
    console.print(
        f"[bold]loaded[/] features={feats.shape} splits={splits.shape} "
        f"-> joined={joined.shape}, feature_cols={len(feature_cols)}"
    )
    return joined, feature_cols


def _fit_one_fold(
    fold_id: int,
    train_X: NDArray[np.float32],
    train_y: NDArray[np.float64],
    val_X: NDArray[np.float32],
    val_y: NDArray[np.float64],
    n_estimators: int,
    early_stop: int,
    wandb_run: Any | None,
) -> tuple[lgb.Booster, NDArray[np.float64], dict[str, float | int]]:
    """Train one fold and return (booster, val_pred, metrics)."""
    console.print(f"[bold cyan]fold {fold_id}[/] train={train_X.shape} val={val_X.shape}")
    train_ds = lgb.Dataset(train_X, label=train_y, free_raw_data=False)
    val_ds = lgb.Dataset(val_X, label=val_y, reference=train_ds, free_raw_data=False)

    callbacks: list[Any] = [
        lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
        lgb.log_evaluation(period=200),
    ]
    if wandb_run is not None:
        # lazy import so the script runs without wandb installed when --no-wandb is set
        from wandb.integration.lightgbm import wandb_callback

        callbacks.append(wandb_callback())

    t0 = time.monotonic()
    booster = lgb.train(
        LGB_PARAMS,
        train_ds,
        num_boost_round=n_estimators,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        feval=_lgb_amex_eval,
        callbacks=callbacks,
    )
    elapsed = time.monotonic() - t0

    val_pred = _as_f64(booster.predict(val_X, num_iteration=booster.best_iteration))
    metrics = _all_metrics(val_y, val_pred)
    fold_metrics: dict[str, float | int] = {
        "fold": fold_id,
        **metrics,
        "best_iter": int(booster.best_iteration or 0),
        "wall_time_s": elapsed,
    }
    console.print(
        f"  fold {fold_id}: AMEX={metrics['amex']:.4f} "
        f"(G={metrics['amex_g']:.4f} D={metrics['amex_d']:.4f}) "
        f"AUC={metrics['auc']:.4f} KS={metrics['ks']:.4f} ll={metrics['log_loss']:.4f} "
        f"best_iter={fold_metrics['best_iter']} {elapsed:.1f}s"
    )

    if wandb_run is not None:
        wandb_run.log({f"fold_{fold_id}/{k}": v for k, v in fold_metrics.items()})

    return booster, val_pred, fold_metrics


def _init_wandb(
    *,
    enabled: bool,
    n_features: int,
    n_trainval: int,
    n_test: int,
    n_estimators: int,
    early_stop: int,
) -> Any | None:
    if not enabled:
        return None
    import wandb

    return wandb.init(
        project=WANDB_PROJECT,
        tags=["baseline", "lgbm", "phase1"],
        config={
            **LGB_PARAMS,
            "n_estimators": n_estimators,
            "early_stop": early_stop,
            "n_features": n_features,
            "n_trainval": n_trainval,
            "n_test": n_test,
            "splits_version": "v1",
        },
        reinit=True,
    )


def _persist(
    *,
    oof_path: Path,
    metrics_path: Path,
    summary: dict[str, Any],
    ids: list[str],
    oof_pred: NDArray[np.float64],
    y: NDArray[np.float64],
    fold: NDArray[np.int64],
) -> None:
    oof_df = pl.DataFrame({KEY_COL: ids, "prediction": oof_pred, TARGET_COL: y, FOLD_COL: fold})
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_df.write_parquet(oof_path, compression="zstd")
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    console.print(f"[bold green]OOF written:[/] {oof_path}")
    console.print(f"[bold green]metrics written:[/] {metrics_path}")


@app.command()
def main(  # noqa: PLR0915
    features_path: Path = typer.Option(
        Path("data/processed/v1/features_lgbm.parquet"),
        "--features",
        help="Customer-level engineered features (output of feature_engineering).",
    ),
    splits_path: Path = typer.Option(
        Path("data/splits/v1.parquet"),
        "--splits",
        help="Canonical splits parquet with [customer_ID, target, split, fold].",
    ),
    oof_path: Path = typer.Option(
        Path("data/processed/v1/oof_lgbm.parquet"),
        "--oof-out",
        help="Where to write per-customer OOF predictions.",
    ),
    metrics_path: Path = typer.Option(
        Path("data/processed/v1/oof_lgbm_metrics.json"),
        "--metrics-out",
        help="Per-fold + aggregated metrics JSON.",
    ),
    wandb_enabled: bool = typer.Option(
        True,
        "--wandb/--no-wandb",
        help="Log training to Weights & Biases (project amex-ssl, tag baseline).",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Smoke-test: 1 fold, 500 trees only.",
    ),
    subset_fraction: float = typer.Option(
        1.0,
        "--subset-fraction",
        min=0.001,
        max=1.0,
        help="Few-shot: train on a stratified subset of trainval customers.",
    ),
    subset_seed: int = typer.Option(1234, "--subset-seed"),
) -> None:
    """Train 5-fold LightGBM baseline and dump OOF predictions + metrics."""
    joined, feature_cols = _load_features_and_splits(features_path, splits_path)

    trainval = joined.filter(pl.col(SPLIT_COL) != TEST_SPLIT)
    test = joined.filter(pl.col(SPLIT_COL) == TEST_SPLIT)

    # Optional few-shot subset of the labeled trainval pool.
    if subset_fraction < 1.0:
        from amex.data.subset import stratified_trainval_subset

        subset_ids = stratified_trainval_subset(splits_path, subset_fraction, seed=subset_seed)
        before = trainval.height
        trainval = trainval.filter(pl.col(KEY_COL).is_in(list(subset_ids)))
        console.print(
            f"[yellow]few-shot subset[/] fraction={subset_fraction} "
            f"-> {trainval.height:,} / {before:,} trainval customers"
        )
        # The default min_child_samples=2400 was tuned for the full dataset; on
        # small few-shot subsets a tree can't ever split with that constraint.
        # Cap it so each leaf still gets >=20 rows but never more than 5% of
        # the per-fold training pool.
        approx_per_fold = max(20, trainval.height * 4 // 5 // 20)
        LGB_PARAMS["min_child_samples"] = min(LGB_PARAMS["min_child_samples"], approx_per_fold)
        console.print(f"  -> min_child_samples auto-scaled to {LGB_PARAMS['min_child_samples']}")

    folds = sorted(set(trainval[FOLD_COL].to_list()) - {TEST_FOLD_SENTINEL})
    n_estimators, early_stop = (500, 50) if quick else (N_ESTIMATORS, EARLY_STOP)
    if quick:
        folds = folds[:1]

    wandb_run = _init_wandb(
        enabled=wandb_enabled,
        n_features=len(feature_cols),
        n_trainval=trainval.height,
        n_test=test.height,
        n_estimators=n_estimators,
        early_stop=early_stop,
    )

    X_all = trainval.select(feature_cols).to_numpy().astype(np.float32)
    y_all = trainval[TARGET_COL].to_numpy().astype(np.float64)
    fold_all = trainval[FOLD_COL].to_numpy().astype(np.int64)
    ids_all = trainval[KEY_COL].to_list()

    oof_pred = np.full(trainval.height, np.nan, dtype=np.float64)
    test_X = test.select(feature_cols).to_numpy().astype(np.float32)
    test_pred_accum = np.zeros(test.height, dtype=np.float64)

    fold_metrics_list: list[dict[str, float | int]] = []
    for fold_id in folds:
        val_mask = fold_all == fold_id
        train_mask = ~val_mask
        booster, val_pred, fold_metrics = _fit_one_fold(
            fold_id,
            X_all[train_mask],
            y_all[train_mask],
            X_all[val_mask],
            y_all[val_mask],
            n_estimators=n_estimators,
            early_stop=early_stop,
            wandb_run=wandb_run,
        )
        oof_pred[val_mask] = val_pred
        if test.height > 0:
            test_pred_accum += _as_f64(
                booster.predict(test_X, num_iteration=booster.best_iteration)
            )
        fold_metrics_list.append(fold_metrics)

    valid_oof_mask = ~np.isnan(oof_pred)
    oof_metrics = _all_metrics(y_all[valid_oof_mask], oof_pred[valid_oof_mask]) | {
        "n": int(valid_oof_mask.sum())
    }

    test_metrics: dict[str, float | int] = {}
    if test.height > 0 and len(folds) > 0:
        test_y = test[TARGET_COL].to_numpy().astype(np.float64)
        test_pred = test_pred_accum / len(folds)
        test_metrics = cast("dict[str, float | int]", _all_metrics(test_y, test_pred))
        test_metrics["n"] = test.height

    summary = {
        "folds": fold_metrics_list,
        "oof": oof_metrics,
        "test": test_metrics,
        "splits_version": "v1",
        "lgb_params": LGB_PARAMS,
        "n_estimators": n_estimators,
        "early_stop": early_stop,
        "quick": quick,
    }

    summary_table = Table(title="LightGBM baseline", show_lines=False)
    summary_table.add_column("metric", style="cyan")
    summary_table.add_column("OOF (trainval)", justify="right")
    if test_metrics:
        summary_table.add_column("test (bagged)", justify="right")
    for key in ("amex", "amex_g", "amex_d", "auc", "ks", "log_loss"):
        row = [key, f"{float(oof_metrics[key]):.5f}"]
        if test_metrics:
            row.append(f"{float(test_metrics[key]):.5f}")
        summary_table.add_row(*row)
    console.print(summary_table)

    _persist(
        oof_path=oof_path,
        metrics_path=metrics_path,
        summary=summary,
        ids=ids_all,
        oof_pred=oof_pred,
        y=y_all,
        fold=fold_all,
    )

    if wandb_run is not None:
        wandb_run.log(
            {f"oof/{k}": float(v) for k, v in oof_metrics.items() if isinstance(v, int | float)}
        )
        if test_metrics:
            wandb_run.log(
                {
                    f"test/{k}": float(v)
                    for k, v in test_metrics.items()
                    if isinstance(v, int | float)
                }
            )
        wandb_run.finish()


if __name__ == "__main__":
    app()
