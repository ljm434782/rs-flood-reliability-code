"""Build descriptive post-test calibration-state supplement from frozen JSON outputs."""

import csv
import hashlib
import json
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "outputs" / "publication_evidence_20260928"
TABLE = ROOT / "paper" / "tables" / "calibration_state_interaction.tex"
FIG_PDF = ROOT / "paper" / "figures" / "calibration_state_interaction.pdf"
FIG_PNG = ROOT / "paper" / "figures" / "calibration_state_interaction.png"
PROVENANCE = EVIDENCE / "provenance.json"
MODELS = ("B2", "B4", "B5", "P")
METRICS = ("nll", "brier", "ece15")
LABELS = {"nll": "NLL", "brier": "Brier", "ece15": "ECE"}
COLORS = {"B2": "#666666", "B4": "#D55E00", "B5": "#009E73", "P": "#CC79A7"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows():
    with (EVIDENCE / "paired_effects.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    wanted = {}
    for row in rows:
        if row["comparison"] in {"100_scaled_minus_raw", "0_scaled_minus_raw",
                                  "zero_minus_full_calibration_effect"}:
            wanted[(row["model"], row["comparison"], row["metric"])] = row
    for model in MODELS:
        for metric in METRICS:
            for comparison in ("100_scaled_minus_raw", "0_scaled_minus_raw",
                               "zero_minus_full_calibration_effect"):
                if (model, comparison, metric) not in wanted:
                    raise RuntimeError(f"Missing frozen evidence row: {model}/{comparison}/{metric}")
    return wanted


def f(value):
    return f"{float(value):+.4f}"


def make_table(rows):
    lines = [
        "% Generated from outputs/publication_evidence_20260928/paired_effects.csv.",
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Post-test descriptive change in probability scores after applying the single frozen temperature. Each $d(s)$ is scaled minus raw at the stated S2 condition; negative values improve the score. $I=d(0)-d(100)$ is the optical-state interaction, so positive values indicate a smaller score improvement at zero S2. Brackets are 95\% paired-process bootstrap percentile intervals; $n=19$ processes for every row. NLL and Brier are probability scores and ECE is a calibration summary; none is interpreted as a pure calibration error.}",
        r"\label{tab:calibrationinteraction}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & Score & $d(100)$ & $d(0)$ & $I=d(0)-d(100)$; $I>0$ processes \\",
        r"\midrule",
    ]
    for model in MODELS:
        for metric in METRICS:
            full = rows[(model, "100_scaled_minus_raw", metric)]
            zero = rows[(model, "0_scaled_minus_raw", metric)]
            interaction = rows[(model, "zero_minus_full_calibration_effect", metric)]
            ci = f"[{float(interaction['ci95_low']):+.4f}, {float(interaction['ci95_high']):+.4f}]"
            positive = int(interaction["positive_processes"])
            lines.append(
                f"{model} & {LABELS[metric]} & {f(full['mean_difference'])} & "
                f"{f(zero['mean_difference'])} & {f(interaction['mean_difference'])} {ci}; {positive}/19 " + r"\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    TABLE.write_text("\n".join(lines), encoding="utf-8")


def make_figure(rows):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 120, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), sharey=True)
    ypos = np.arange(len(MODELS))[::-1]
    for ax, metric in zip(axes, METRICS):
        for y, model in zip(ypos, MODELS):
            row = rows[(model, "zero_minus_full_calibration_effect", metric)]
            value = float(row["mean_difference"])
            lo = float(row["ci95_low"])
            hi = float(row["ci95_high"])
            ax.errorbar(value, y, xerr=[[value - lo], [hi - value]], fmt="o",
                        color=COLORS[model], markersize=4, capsize=2, linewidth=1.1)
            ax.text(0.98, y, f"{int(row['positive_processes'])}/19 +",
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7, color=COLORS[model])
        ax.axvline(0, color="#333333", linewidth=.8)
        ax.set_title(LABELS[metric])
        ax.set_xlabel(r"$I=d(0)-d(100)$")
        ax.grid(axis="x", alpha=.2)
        ax.set_yticks(ypos)
    axes[0].set_yticklabels(MODELS)
    fig.suptitle("Frozen-temperature score change across optical states", y=1.02, fontsize=10)
    fig.text(0.5, -0.015, "Positive interaction = less improvement at zero S2; labels show positive process directions",
             ha="center", fontsize=7.5, color="#43515A")
    fig.tight_layout()
    FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PDF, bbox_inches="tight", metadata={"Creator": "frozen v1 evidence supplement"})
    fig.savefig(FIG_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load_rows()
    make_table(rows)
    make_figure(rows)
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    provenance["supplement_outputs"] = {
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": digest(Path(__file__)),
        "table": TABLE.relative_to(ROOT).as_posix(),
        "table_sha256": digest(TABLE),
        "figure_pdf": FIG_PDF.relative_to(ROOT).as_posix(),
        "figure_pdf_sha256": digest(FIG_PDF),
        "figure_png": FIG_PNG.relative_to(ROOT).as_posix(),
        "figure_png_sha256": digest(FIG_PNG),
        "source_rows": "paired_effects.csv; no refitting or new evaluation",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(TABLE)
    print(FIG_PDF)


if __name__ == "__main__":
    main()
