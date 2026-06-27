"""Phase 5-D multi-seed error bar plot.

Shows Phase 1 baseline (grey) and three SSL pretrain seeds + their mean
(orange error bar). Demonstrates that GBM + SSL > GBM is robust across
SSL pretrain randomness.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "figures" / "multiseed_errorbar.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

BASELINE = 0.79558

# Test bagged AMEX per seed (Phase 4 = seed 42 = "original", + Phase 5-D s1, s2)
SEEDS = {
    "seed 42 (orig)": 0.79662,
    "seed 1": 0.79768,
    "seed 2": 0.79669,
}


def main() -> None:
    vals = list(SEEDS.values())
    mu, sigma = mean(vals), stdev(vals)

    fig, ax = plt.subplots(figsize=(7, 4.2))

    # baseline line + label
    ax.axhline(
        BASELINE,
        color="grey",
        linestyle="--",
        linewidth=1.0,
        label=f"Phase-1 GBM only ({BASELINE:.5f})",
    )

    # individual seed points
    xs = list(range(len(SEEDS)))
    ax.scatter(xs, vals, color="#DD8452", s=80, zorder=3, label="individual seed")
    for x, (name, v) in zip(xs, SEEDS.items(), strict=True):
        ax.text(x, v + 0.0003, f"{v:.5f}", ha="center", fontsize=9)
        ax.text(x, BASELINE - 0.0006, name, ha="center", fontsize=8)

    # mean ± std as a wider transparent band on the right
    x_mean = len(SEEDS)
    ax.errorbar(
        [x_mean],
        [mu],
        yerr=[sigma],
        fmt="o",
        capsize=10,
        color="#4C72B0",
        markersize=10,
        elinewidth=2,
        label=f"mean ± std ({mu:.5f} ± {sigma:.5f})",
    )
    ax.text(x_mean, mu + 0.0003, f"{mu:.5f}\n±{sigma:.5f}", ha="center", fontsize=9)
    ax.text(x_mean, BASELINE - 0.0006, "mean of 3", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks([])
    ax.set_xlim(-0.5, len(SEEDS) + 0.5)
    ax.set_ylim(BASELINE - 0.001, max(vals) + 0.0009)
    ax.set_ylabel("Test bagged AMEX (n = 45,892)")
    ax.set_title("Phase 5-D: GBM + 128 SSL hybrid emb across 3 SSL pretrain seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    delta = mu - BASELINE
    ax.annotate(
        f"Δ = +{delta:.5f}\nt ≈ {delta / (sigma / len(vals) ** 0.5):.2f}, df=2",
        xy=(x_mean - 1.0, (BASELINE + mu) / 2),
        ha="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "grey"},
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")

    summary = {
        "baseline": BASELINE,
        "seeds": SEEDS,
        "mean": mu,
        "std": sigma,
        "delta_mean": delta,
        "t_stat": delta / (sigma / len(vals) ** 0.5),
        "df": len(vals) - 1,
    }
    (OUT.parent / "multiseed_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
