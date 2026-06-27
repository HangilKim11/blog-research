"""Generate the few-shot label-efficiency curve and write it to reports/figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FEW_SHOT_DIR = ROOT / "data" / "processed" / "v1" / "few_shot"
OUT_DIR = ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRACTIONS = [0.01, 0.05, 0.25, 1.0]


def _load(name: str, fraction: float) -> dict[str, float]:
    payload = json.loads((FEW_SHOT_DIR / f"{name}_f{fraction}.json").read_text("utf-8"))
    return payload["test"]


def main() -> None:
    gbm = [_load("gbm", f)["amex"] for f in FRACTIONS]
    ssl = [_load("ssl", f)["amex"] for f in FRACTIONS]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(FRACTIONS, gbm, marker="o", color="C0", label="LightGBM (engineered features)")
    ax.plot(FRACTIONS, ssl, marker="s", color="C1", label="SSL hybrid + full fine-tune")
    ax.axhline(
        0.79558,
        color="grey",
        linestyle="--",
        linewidth=0.8,
        label="Phase-1 GBM full data (0.79558)",
    )
    ax.set_xscale("log")
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in FRACTIONS])
    ax.set_xlabel("Labeled fraction of train+val")
    ax.set_ylabel("Test AMEX metric")
    ax.set_title("Label-efficiency: GBM vs SSL fine-tune (same stratified subset)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # annotate
    for f, g, s in zip(FRACTIONS, gbm, ssl, strict=True):
        ax.annotate(
            f"{g:.3f}", (f, g), xytext=(5, 5), textcoords="offset points", color="C0", fontsize=8
        )
        ax.annotate(
            f"{s:.3f}", (f, s), xytext=(5, -12), textcoords="offset points", color="C1", fontsize=8
        )

    out = OUT_DIR / "few_shot_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
