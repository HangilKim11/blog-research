"""Validate the local environment for the current research phase.

Phase 1 (baseline): needs Python 3.11+, polars, lightgbm, kaggle.json.
Phase 2 (SSL):       additionally needs torch with CUDA available.

Run
---
    uv run python scripts/check_env.py
    uv run python scripts/check_env.py --phase 2
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


def _check(label: str, ok: bool, detail: str = "") -> tuple[str, str, str]:
    status = "[green]OK[/]" if ok else "[red]FAIL[/]"
    return label, status, detail


def _check_module(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
    except ImportError as e:
        return False, f"import error: {e}"
    version = getattr(mod, "__version__", "unknown")
    return True, f"v{version}"


@app.command()
def main(
    phase: int = typer.Option(1, "--phase", help="Which phase's requirements to check (1 or 2)."),
) -> None:
    """Run a phase-aware environment check."""
    rows: list[tuple[str, str, str]] = []
    failures = 0

    # Python
    py_ok = sys.version_info >= (3, 11)
    rows.append(_check("Python >= 3.11", py_ok, f"{sys.version.split()[0]}"))
    failures += not py_ok

    # Core libs (both phases)
    for pkg in ("polars", "lightgbm", "wandb", "kaggle"):
        ok, detail = _check_module(pkg)
        rows.append(_check(pkg, ok, detail))
        failures += not ok

    # Kaggle credentials
    home = Path(os.environ.get("USERPROFILE") or Path.home())
    kj = home / ".kaggle" / "kaggle.json"
    rows.append(_check(f"kaggle.json @ {kj}", kj.is_file(), ""))
    failures += not kj.is_file()

    if phase >= 2:
        # Deep learning extras
        for pkg in ("torch", "pytorch_lightning", "transformers", "accelerate"):
            ok, detail = _check_module(pkg)
            rows.append(_check(pkg, ok, detail))
            failures += not ok

        # CUDA
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                props = torch.cuda.get_device_properties(0)
                detail = f"{props.name}, {props.total_memory / 1024**3:.1f} GiB, sm_{props.major}{props.minor}"
            else:
                detail = "torch installed but CUDA not visible"
            rows.append(_check("torch.cuda.is_available()", cuda_ok, detail))
            failures += not cuda_ok
        except ImportError:
            rows.append(_check("torch.cuda check", False, "torch not importable"))
            failures += 1

    table = Table(title=f"Environment check -- phase {phase}", show_lines=False)
    table.add_column("check", style="cyan")
    table.add_column("status", justify="center")
    table.add_column("detail")
    for label, status, detail in rows:
        table.add_row(label, status, detail)
    console.print(table)

    if failures:
        console.print(f"\n[red]{failures} check(s) failed.[/]")
        raise typer.Exit(code=1)
    console.print("\n[green]all checks passed.[/]")


if __name__ == "__main__":
    app()
