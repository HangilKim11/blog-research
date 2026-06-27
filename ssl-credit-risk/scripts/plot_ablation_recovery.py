"""Phase 5 ablation step diagram: how much of the top-100 loss does SSL recover?

Produces a bar chart with three bars (i) full hand, (ii) hand - top100,
(iii) (hand - top100) + SSL, with arrows annotating the lost signal and
the SSL-recovered fraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "figures" / "ablation_recovery.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# values from data/processed/v1/oof_lgbm_*_metrics.json
HAND = 0.79558
MINUS = 0.78966
MINUS_PLUS_SSL = 0.79290
FULL_PLUS_SSL = 0.79662


def main() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = [
        "(i)\nfull hand\n(1,291)",
        "(ii)\nhand - top-100\n(1,191)",
        "(iii)\n(hand - top-100)\n+ SSL (1,319)",
    ]
    values = [HAND, MINUS, MINUS_PLUS_SSL]
    colors = ["#4C72B0", "#C44E52", "#DD8452"]

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0005, f"{v:.5f}", ha="center", fontsize=10)

    ax.set_ylim(0.785, 0.798)
    ax.set_ylabel("Test AMEX (n = 45,892)")
    ax.set_title("Phase 5-A ablation: SSL recovers ~55% of top-100 hand-feature signal")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate the lost and recovered amounts.
    ax.annotate(
        f"lost\n-{HAND - MINUS:.5f}",
        xy=(0.5, (HAND + MINUS) / 2),
        xytext=(0.5, (HAND + MINUS) / 2),
        ha="center",
        fontsize=9,
        color="#C44E52",
    )
    ax.annotate(
        f"recovered\n+{MINUS_PLUS_SSL - MINUS:.5f}\n({(MINUS_PLUS_SSL - MINUS) / (HAND - MINUS) * 100:.1f}% of loss)",
        xy=(1.5, (MINUS + MINUS_PLUS_SSL) / 2),
        xytext=(1.5, (MINUS + MINUS_PLUS_SSL) / 2),
        ha="center",
        fontsize=9,
        color="#DD8452",
    )

    # Reference line for Phase 4 (full hand + SSL).
    ax.axhline(FULL_PLUS_SSL, color="grey", linestyle=":", linewidth=0.8)
    ax.text(
        2.45,
        FULL_PLUS_SSL,
        f" Phase 4 (full+SSL)\n {FULL_PLUS_SSL:.5f}",
        va="center",
        fontsize=8,
        color="grey",
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")

    json_out = OUT.parent / "ablation_recovery.json"
    json_out.write_text(
        json.dumps(
            {
                "hand_only": HAND,
                "hand_minus_top100": MINUS,
                "hand_minus_top100_plus_ssl": MINUS_PLUS_SSL,
                "full_hand_plus_ssl": FULL_PLUS_SSL,
                "lost": HAND - MINUS,
                "recovered": MINUS_PLUS_SSL - MINUS,
                "recovery_rate": (MINUS_PLUS_SSL - MINUS) / (HAND - MINUS),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {json_out}")


if __name__ == "__main__":
    main()
