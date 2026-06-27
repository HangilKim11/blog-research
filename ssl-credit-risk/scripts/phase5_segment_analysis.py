"""Segment-level AMEX comparison: GBM-only vs GBM+SSL on test customers.

Reads OOF parquets from Phase 1 (hand-only) and Phase 4 (hand+SSL), joins
with the held-out test predictions reconstructed from the metrics JSON,
then slices the test set by customer characteristics and reports per-segment
AMEX delta.

Segments
--------
- statement count (thin file vs rich file)
- P_2_last decile (balance proxy)
- target prior in segment (does SSL help where GBM is uncertain?)

Output: reports/figures/segment_decomposition.png + reports/phase5_segments.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from amex.evaluation.metrics import amex_metric_components

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=False)

KEY_COL = "customer_ID"
TARGET_COL = "target"


def _load_test_preds(
    *,
    splits_path: Path,
    features_lgbm: Path,
    features_aug: Path,
    oof_lgbm_metrics: Path,
    oof_aug_metrics: Path,
) -> pl.DataFrame:
    """Reconstruct per-customer test predictions for both models.

    The lgbm.py script bags test predictions across folds; the JSON we wrote
    only stores the aggregated metric, not per-row predictions. So we instead
    re-rank the OOF parquet for trainval and use the trainval-on-fold-0 model
    is overkill here. Instead, we recompute by reading the OOF for trainval
    (Phase 1 vs Phase 4) and bag the test inside this script.

    Simpler path: this script ASSUMES the user has run both Phase 1 and Phase
    4 trainings, which produce per-customer OOF for trainval. For test rows we
    fall back to reading the test_pred_accum trail -- but lgbm.py doesn't
    persist that. So instead we re-derive test predictions from the per-row
    OOF and the fold structure.

    -- For now we approximate: compare TEST-segment metrics by reading the
    feature parquets and the saved aggregate test metrics. We can extend the
    analysis to per-row test scores if lgbm.py is later modified to save them.
    """
    raise NotImplementedError("placeholder; see _load_test_preds_v2 below")


def _segment_amex(
    frame: pl.DataFrame, *, segment_col: str, pred_col: str
) -> dict[str, dict[str, float]]:
    """Group by ``segment_col`` and compute AMEX inside each segment."""
    out: dict[str, dict[str, float]] = {}
    for seg_val in frame[segment_col].unique().sort().to_list():
        sub = frame.filter(pl.col(segment_col) == seg_val)
        y = sub[TARGET_COL].to_numpy()
        p = sub[pred_col].to_numpy()
        if y.sum() == 0 or y.sum() == len(y):
            continue  # AMEX undefined for single-class segments
        m, g, d = amex_metric_components(y, p)
        out[str(seg_val)] = {
            "n": len(y),
            "n_default": int(y.sum()),
            "default_rate": float(y.mean()),
            "amex": float(m),
            "amex_g": float(g),
            "amex_d": float(d),
        }
    return out


@app.command()
def main(  # noqa: PLR0915 -- linear analysis script
    oof_baseline: Path = typer.Option(Path("data/processed/v1/oof_lgbm.parquet"), "--oof-baseline"),
    oof_augmented: Path = typer.Option(
        Path("data/processed/v1/oof_lgbm_augmented.parquet"), "--oof-aug"
    ),
    splits_path: Path = typer.Option(Path("data/splits/v1.parquet"), "--splits"),
    features_lgbm: Path = typer.Option(
        Path("data/processed/v1/features_lgbm.parquet"), "--features-lgbm"
    ),
    out_json: Path = typer.Option(Path("reports/phase5_segments.json"), "--out-json"),
    out_png: Path = typer.Option(Path("reports/figures/segment_decomposition.png"), "--out-png"),
) -> None:
    """Compute per-segment AMEX deltas between Phase 1 and Phase 4 OOF.

    This analyses TRAINVAL (OOF) since per-customer predictions exist there.
    Conclusions transfer to test under the iid assumption we've operated under
    throughout the project.
    """
    base = pl.read_parquet(oof_baseline).rename({"prediction": "pred_base"})
    aug = pl.read_parquet(oof_augmented).rename({"prediction": "pred_aug"})
    frame = base.join(aug.select([KEY_COL, "pred_aug"]), on=KEY_COL)

    # Segment 1: number of statements per customer (thin vs rich file).
    feats = pl.read_parquet(features_lgbm)
    if "n_statements" in feats.columns:
        frame = frame.join(feats.select([KEY_COL, "n_statements"]), on=KEY_COL)
        frame = frame.with_columns(
            pl.when(pl.col("n_statements") <= 6)
            .then(pl.lit("thin (<=6 stmts)"))
            .when(pl.col("n_statements") <= 11)
            .then(pl.lit("mid (7-11)"))
            .otherwise(pl.lit("rich (12-13)"))
            .alias("statements_bin")
        )
    else:
        console.print("[yellow]n_statements not in features parquet -- skipping bin[/]")

    # Segment 2: P_2_last decile (balance/utilization proxy).
    if "P_2_last" in feats.columns:
        frame = frame.join(feats.select([KEY_COL, "P_2_last"]), on=KEY_COL)
        frame = frame.with_columns(
            pl.col("P_2_last")
            .rank("average")
            .truediv(frame.height)
            .mul(10)
            .floor()
            .clip(0, 9)
            .cast(pl.Int8)
            .alias("p2_decile")
        )
    else:
        console.print("[yellow]P_2_last not in features parquet -- skipping decile[/]")

    # Segment 3: base prediction decile (where is GBM uncertain?).
    frame = frame.with_columns(
        pl.col("pred_base")
        .rank("average")
        .truediv(frame.height)
        .mul(10)
        .floor()
        .clip(0, 9)
        .cast(pl.Int8)
        .alias("base_pred_decile")
    )

    # ---- compute per-segment AMEX ----
    segments_data: dict[str, dict[str, dict[str, float | int]]] = {}
    for seg_col in ("statements_bin", "p2_decile", "base_pred_decile"):
        if seg_col not in frame.columns:
            continue
        base_seg = _segment_amex(frame, segment_col=seg_col, pred_col="pred_base")
        aug_seg = _segment_amex(frame, segment_col=seg_col, pred_col="pred_aug")
        # combine
        combined: dict[str, dict[str, float | int]] = {}
        for k in base_seg:
            b = base_seg[k]
            a = aug_seg.get(k, b)
            combined[k] = {
                "n": b["n"],
                "n_default": b["n_default"],
                "default_rate": b["default_rate"],
                "amex_base": b["amex"],
                "amex_aug": a["amex"],
                "delta": a["amex"] - b["amex"],
            }
        segments_data[seg_col] = combined

    # ---- print + save ----
    for seg_col, segs in segments_data.items():
        table = Table(title=f"Segment {seg_col}", show_lines=False)
        table.add_column("segment", style="cyan")
        table.add_column("n", justify="right")
        table.add_column("default rate", justify="right")
        table.add_column("base AMEX", justify="right")
        table.add_column("aug AMEX", justify="right")
        table.add_column("delta", justify="right")
        for k in sorted(segs.keys()):
            s = segs[k]
            sign = "+" if s["delta"] >= 0 else ""
            table.add_row(
                k,
                f"{s['n']:,}",
                f"{s['default_rate']:.3f}",
                f"{s['amex_base']:.4f}",
                f"{s['amex_aug']:.4f}",
                f"{sign}{s['delta']:+.4f}",
            )
        console.print(table)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(segments_data, indent=2) + "\n", encoding="utf-8")
    console.print(f"[bold green]wrote[/] {out_json}")

    # ---- figure ----
    fig, axes = plt.subplots(1, len(segments_data), figsize=(5 * len(segments_data), 4))
    if len(segments_data) == 1:
        axes = [axes]
    for ax, (seg_col, segs) in zip(axes, segments_data.items(), strict=False):
        keys = sorted(segs.keys())
        base_vals = [segs[k]["amex_base"] for k in keys]
        aug_vals = [segs[k]["amex_aug"] for k in keys]
        x = np.arange(len(keys))
        width = 0.4
        ax.bar(x - width / 2, base_vals, width, label="GBM hand only", color="C0")
        ax.bar(x + width / 2, aug_vals, width, label="GBM + SSL emb", color="C1")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in keys], rotation=45 if "_" in seg_col else 0, fontsize=8)
        ax.set_xlabel(seg_col)
        ax.set_ylabel("OOF AMEX")
        ax.set_title(seg_col)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Phase 5 segment decomposition: where does +0.001 live?")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    console.print(f"[bold green]wrote[/] {out_png}")


if __name__ == "__main__":
    app()
