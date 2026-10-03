"""Build the compact B4 classwise contribution supplement."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/robustness_v6"
TABLE = ROOT / "paper/tables/v6_category_contributions.tex"
FIG = ROOT / "paper/figures/v6_category_contributions.pdf"


def fmt(x):
    return f"{float(x):+.4f}"


def main():
    rows = json.loads((BASE / "v6_category_interactions.json").read_text(encoding="utf-8"))
    b4 = [r for r in rows if r["candidate"] == "B4_v6"]
    labels = {0: "Non-water", 1: "Permanent water", 2: "Flood"}
    lines = [r"\begin{table}[H]",
             r"\caption{B4 classwise contribution to the exploratory v6 interaction. Positive values mean that the fixed temperature changes the score less at zero S2 than at full S2. Intervals are 95\% process-bootstrap percentile intervals; the defined count is the number of development processes with that true class in both endpoint states.}",
             r"\label{tab:v6category}", r"\centering\scriptsize",
             r"\begin{tabular}{rllrr}", r"Seed & True class & Metric & $I$ & 95\% interval \\", r"\hline"]
    for seed in (20261011, 20261012):
        for c in (0, 1, 2):
            for metric in ("nll", "brier"):
                r = next(x for x in b4 if x["seed"] == seed and x["class"] == c and x["metric"] == metric)
                lines.append(f"{seed} & {labels[c]} & {metric.upper()} & {fmt(r['mean'])} & "
                             f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}] ({r['defined_processes']}) " + r"\\")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=False)
    colors = {0: "#777777", 1: "#4f8a5b", 2: "#c66a2b"}
    for ax, metric in zip(axes, ("nll", "brier")):
        x = np.arange(3)
        for j, seed in enumerate((20261011, 20261012)):
            values = []; lows = []; highs = []
            for c in (0, 1, 2):
                r = next(xr for xr in b4 if xr["seed"] == seed and xr["class"] == c and xr["metric"] == metric)
                values.append(float(r["mean"])); lows.append(float(r["ci95_low"])); highs.append(float(r["ci95_high"]))
            xpos = x + (-.08 if j == 0 else .08)
            ax.errorbar(xpos, values, yerr=[np.asarray(values) - lows, np.asarray(highs) - values],
                        fmt="o", ms=3, capsize=2, label=str(seed))
        ax.axhline(0, color="#333", lw=.7)
        ax.set_title(metric.upper())
        ax.set_xticks(x, ["Non-water", "Permanent\nwater", "Flood"], fontsize=7)
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Classwise I = d(0) − d(100)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("B4 v6 classwise probability contribution", fontsize=9)
    fig.tight_layout(pad=.8)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {TABLE} and {FIG}")


if __name__ == "__main__":
    main()
