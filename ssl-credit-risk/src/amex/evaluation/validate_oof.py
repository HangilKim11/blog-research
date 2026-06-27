"""Validate the AMEX metric against an external OOF prediction file.

Usage
-----
    uv run python -m amex.evaluation.validate_oof --oof PATH [--expected SCORE]

The OOF file must be a Parquet (or CSV) with at least two columns:
    customer_ID, prediction

Optional ``--expected`` lets you assert the computed score against a known
published leaderboard score (e.g. a Kaggle 1st-place writeup) and fail the
script if the absolute difference exceeds ``--tolerance``.

This is the dedicated external check called out in STARTER_PROMPT Step 5: a
green run on a real 1st-place OOF is the final BLOCKER clearance for the
metric implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from amex.evaluation.metrics import amex_metric_components

KEY_COL = "customer_ID"
PRED_COL = "prediction"
TARGET_COL = "target"

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _load_oof(path: Path) -> pl.DataFrame:
    if not path.exists():
        msg = f"OOF file not found: {path}"
        raise FileNotFoundError(msg)
    oof = pl.read_csv(path) if path.suffix.lower() == ".csv" else pl.read_parquet(path)

    if KEY_COL not in oof.columns:
        msg = f"OOF missing column {KEY_COL!r}; got {oof.columns}"
        raise ValueError(msg)
    # Allow either 'prediction' or common alternatives.
    if PRED_COL not in oof.columns:
        for alt in ("preds", "y_pred", "score", "probability"):
            if alt in oof.columns:
                oof = oof.rename({alt: PRED_COL})
                console.print(f"[yellow]using column {alt!r} as 'prediction'[/]")
                break
        else:
            msg = (
                f"OOF missing prediction column; tried {PRED_COL!r} and aliases. Got {oof.columns}"
            )
            raise ValueError(msg)

    return oof.select([KEY_COL, PRED_COL])


def _load_labels(labels_path: Path) -> pl.DataFrame:
    if not labels_path.exists():
        msg = (
            f"Labels not found at {labels_path}. "
            "Run `python -m amex.data.sequence_builder --mode dev` first."
        )
        raise FileNotFoundError(msg)
    return pl.read_parquet(labels_path).select([KEY_COL, TARGET_COL])


@app.command()
def main(
    oof: Path = typer.Option(..., "--oof", help="OOF Parquet/CSV with customer_ID + prediction."),
    labels: Path = typer.Option(
        Path("data/processed/v1/labels.parquet"),
        "--labels",
        help="Customer-level labels Parquet.",
    ),
    expected: float | None = typer.Option(
        None,
        "--expected",
        help="Expected M score from published reference (e.g. Kaggle LB). "
        "If provided, the script exits non-zero on mismatch.",
    ),
    tolerance: float = typer.Option(
        0.001,
        "--tolerance",
        help="Maximum allowed |actual - expected| difference.",
    ),
) -> None:
    """Compute the AMEX metric on an external OOF and (optionally) verify it."""
    oof_df = _load_oof(oof)
    lab_df = _load_labels(labels)

    joined = lab_df.join(oof_df, on=KEY_COL, how="inner")
    matched = joined.height
    missing = lab_df.height - matched
    extra = oof_df.height - matched

    table = Table(title="OOF coverage", show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("labels (total)", f"{lab_df.height:,}")
    table.add_row("oof rows (total)", f"{oof_df.height:,}")
    table.add_row("joined (matched)", f"{matched:,}")
    table.add_row("labels not in OOF", f"{missing:,}")
    table.add_row("OOF not in labels", f"{extra:,}")
    console.print(table)

    if matched == 0:
        console.print("[red]no overlap between OOF and labels -- abort[/]")
        raise typer.Exit(code=2)
    if matched < lab_df.height * 0.95:
        console.print(
            f"[yellow]warning: OOF covers only {matched / lab_df.height:.1%} of labels[/]"
        )

    y_true = joined[TARGET_COL].to_numpy()
    y_pred = joined[PRED_COL].to_numpy()
    m, g, d = amex_metric_components(y_true, y_pred)

    score_table = Table(title="AMEX metric", show_lines=False)
    score_table.add_column("component", style="cyan")
    score_table.add_column("value", justify="right")
    score_table.add_row("M (final)", f"{m:.6f}")
    score_table.add_row("G (norm. Gini)", f"{g:.6f}")
    score_table.add_row("D (top-4% capt.)", f"{d:.6f}")
    console.print(score_table)

    if expected is not None:
        diff = abs(m - expected)
        ok = diff <= tolerance
        verdict = "[green]PASS[/]" if ok else "[red]FAIL[/]"
        console.print(f"{verdict} |M - expected| = {diff:.6f}  (tolerance {tolerance})")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    app()
