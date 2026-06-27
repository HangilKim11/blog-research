"""One-off CLI to fit the SSL tokenizer on the train split and save to disk."""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from amex.ssl.tokenizer import fit_tokenizer

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


@app.command()
def main(
    train_glob: str = typer.Option(
        "data/processed/v1/train/**/*.parquet",
        "--train",
        help="glob for the train partition parquet files.",
    ),
    splits_path: Path = typer.Option(
        Path("data/splits/v1.parquet"),
        "--splits",
        help="canonical splits parquet (we fit on the train split only).",
    ),
    out_path: Path = typer.Option(
        Path("data/processed/v1/tokenizer.json"),
        "--out",
        help="where to save the fitted tokenizer JSON.",
    ),
) -> None:
    """Fit per-feature stats on the train split and write the tokenizer JSON."""
    t0 = time.monotonic()
    tok = fit_tokenizer(train_glob, splits_path)
    elapsed = time.monotonic() - t0

    tok.save(out_path)

    table = Table(title="Tokenizer fit", show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("numeric cols", str(tok.n_numeric))
    table.add_row("categorical cols", str(tok.n_categorical))
    cat_sizes = [tok.cat_vocab_size(c) for c in tok.categorical_cols]
    table.add_row("cat vocab sizes (min..max)", f"{min(cat_sizes)}..{max(cat_sizes)}")
    table.add_row("wall time", f"{elapsed:,.1f} s")
    console.print(table)
    console.print(f"[bold green]tokenizer written:[/] {out_path}")


if __name__ == "__main__":
    app()
