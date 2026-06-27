"""Join hand-crafted GBM features with SSL encoder embeddings.

Reads:
  - data/processed/v1/features_lgbm.parquet  (1,293 cols: customer_ID + 1291 features + target)
  - data/processed/v1/features_ssl_*.parquet (customer_ID + emb_NNN cols)

Writes a single parquet with all columns joined on customer_ID:
  data/processed/v1/features_augmented.parquet

The joined frame is what the LightGBM trainer consumes via --features.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=False)

KEY_COL = "customer_ID"


@app.command()
def main(
    base: Path = typer.Option(
        Path("data/processed/v1/features_lgbm.parquet"),
        "--base",
        help="hand-crafted GBM features parquet.",
    ),
    ssl: list[Path] = typer.Option(
        ...,
        "--ssl",
        help="one or more SSL embedding parquets to join (customer_ID + emb cols).",
    ),
    out: Path = typer.Option(
        Path("data/processed/v1/features_augmented.parquet"),
        "--out",
    ),
) -> None:
    """Inner-join base features with one or more SSL embedding parquets."""
    frame = pl.read_parquet(base)
    console.print(f"[bold]base[/]    {frame.shape}  cols={len(frame.columns)}")

    for s in ssl:
        if not s.exists():
            msg = f"SSL features parquet not found: {s}"
            raise FileNotFoundError(msg)
        emb = pl.read_parquet(s)
        before_cols = len(frame.columns)
        frame = frame.join(emb, on=KEY_COL, how="left")
        added = len(frame.columns) - before_cols
        # Sanity: every row should have matched (left-join + complete embed cover).
        missing = frame.select(pl.col(emb.columns[1]).is_null().sum()).item()
        console.print(
            f"[bold]+ {s.name}[/] {emb.shape}  joined +{added} cols  null-after-join={missing}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out, compression="zstd")

    table = Table(title="Augmented features", show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("rows (customers)", f"{frame.height:,}")
    table.add_row("total columns", f"{len(frame.columns):,}")
    table.add_row("file size", f"{out.stat().st_size / 1024**2:,.1f} MB")
    console.print(table)
    console.print(f"[bold green]wrote[/] {out}")


if __name__ == "__main__":
    app()
