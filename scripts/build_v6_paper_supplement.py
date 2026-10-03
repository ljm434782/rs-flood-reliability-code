"""Build the v6 exploratory interaction table and figure from frozen outputs."""

import csv
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/robustness_v6"
TABLE = ROOT / "paper/tables/v6_robustness_interaction.tex"
FIG = ROOT / "paper/figures/v6_robustness_interaction.pdf"
CSV = BASE / "v6_interaction.csv"


def read_rows():
    with CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value):
    return "--" if value in (None, "", "nan") else f"{float(value):+.4f}"


def main():
    rows = read_rows()
    display_metric = {"nll": "NLL", "brier": "Brier", "ece15": "ECE15", "flood_iou": "Flood IoU"}
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[H]",
        r"\caption{Exploratory v6 process-paired interaction on the training-derived development fold. $I=d(0)-d(100)$, where $d$ is scaled minus raw after one temperature fitted on the 65-process fit pool. Positive values mean a smaller score benefit, or a larger adverse change, at zero S2. Intervals are 95\% process-bootstrap percentile intervals (10,000 resamples), conditional on checkpoint selection and the fixed training-source temperature.}",
        r"\label{tab:v6interaction}",
        r"\centering\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"Model & Seed & Metric & $I$ & 95\% interval & $+/n$ \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(f"{row['candidate'].replace('_v6','')} & {row['seed']} & {display_metric[row['metric']]} & "
                     f"{fmt(row['mean'])} & [{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] & "
                     f"{row['positive_processes']}/{row['process_count']} " + r"\\")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    TABLE.write_text("\n".join(lines), encoding="utf-8")

    metrics = ["nll", "brier", "ece15"]
    candidates = ["B4_v6", "B5_v6", "P_v6"]
    seed_styles = {"20261011": (-0.10, "o"), "20261012": (0.10, "s")}
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.7), sharey=False)
    colors = {"B4_v6": "#1965a0", "B5_v6": "#4f8a5b", "P_v6": "#c66a2b"}
    for ax, metric in zip(axes, metrics):
        for x, candidate in enumerate(candidates):
            subset = [r for r in rows if r["candidate"] == candidate and r["metric"] == metric]
            for row in subset:
                offset, marker = seed_styles[row["seed"]]
                value = float(row["mean"])
                low, high = float(row["ci95_low"]), float(row["ci95_high"])
                ax.errorbar(x + offset, value,
                            yerr=[[value - low], [high - value]], fmt=marker,
                            color=colors[candidate], capsize=2, ms=3)
        ax.axhline(0, color="#333333", lw=.7)
        ax.set_title(metric.upper())
        ax.set_xticks(range(3), ["B4", "B5", "P"])
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("I = d(0) − d(100)")
    axes[0].legend(handles=[Line2D([], [], color="#333333", linestyle="none",
                                  marker=style[1], markersize=3, label=seed)
                           for seed, style in seed_styles.items()],
                   frameon=False, fontsize=6, loc="upper left")
    fig.suptitle("v6 development-fold temperature interaction", fontsize=9)
    fig.tight_layout(pad=.8)
    fig.savefig(FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {TABLE} and {FIG}")


if __name__ == "__main__":
    main()
