"""Make publication figures directly from frozen formal result files."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "formal_v1"
TABLES = BASE / "summary_tables"
OUT = ROOT / "paper" / "figures"
COLORS = {"B1": "#666666", "B2": "#0072B2", "B4": "#D55E00",
          "B5": "#009E73", "P": "#CC79A7"}
PDF_METADATA = {"CreationDate": datetime(2026, 9, 27, tzinfo=timezone.utc),
                "ModDate": datetime(2026, 9, 27, tzinfo=timezone.utc)}


def csv_rows(name):
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def result(model, name):
    return json.loads((BASE / model / f"{name}.json").read_text(encoding="utf-8"))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def set_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelsize": 10, "figure.dpi": 120,
                         "savefig.facecolor": "white"})


def retention_curve():
    rows = csv_rows("optical_retention_curve.csv")
    fig, ax = plt.subplots(figsize=(6.6, 3.65))
    for model in ("B1", "B2", "B4", "B5", "P"):
        items = sorted((row for row in rows if row["model"] == model),
                       key=lambda row: int(row["s2_block_retention_pct"]))
        x = np.asarray([int(row["s2_block_retention_pct"]) for row in items])
        y = np.asarray([float(row["flood_iou"]) for row in items])
        low = np.asarray([float(row["flood_iou_ci_low"]) for row in items])
        high = np.asarray([float(row["flood_iou_ci_high"]) for row in items])
        ax.plot(x, y, color=COLORS[model], marker="o", linewidth=1.8,
                markersize=4, label=model)
        if model in {"B4", "B5", "P"}:
            shift = {"B4": -2, "B5": 0, "P": 2}[model]
            at = np.asarray((0, 100))
            where = np.isin(x, at)
            ax.errorbar(x[where] + shift, y[where],
                        yerr=np.vstack((y[where] - low[where], high[where] - y[where])),
                        fmt="none", ecolor=COLORS[model], capsize=2, alpha=.65,
                        linewidth=.8)
    ax.set(xlim=(-5, 105), ylim=(0.08, 0.43), xticks=(0, 25, 50, 75, 100),
           xlabel="Additional S2 block retention (%)", ylabel="Process-mean flood IoU")
    ax.grid(axis="y", alpha=.22)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.16))
    fig.tight_layout()
    save(fig, "optical_retention")


def calibration_shift():
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.55), sharex=True)
    for col, model in enumerate(("B4", "B5", "P")):
        summary = result(model, "internal_test")["summary"]
        x = np.asarray((0, 25, 50, 75, 100))
        for row_idx, metric in enumerate(("nll", "ece15")):
            ax = axes[row_idx, col]
            for state, linestyle, marker, label in (("raw", "--", "s", "Raw"),
                                                     ("calibrated", "-", "o", "Scaled")):
                y = [summary[f"{level}_{state}"][metric]["mean"] for level in x]
                ax.plot(x, y, color=COLORS[model], linestyle=linestyle,
                        marker=marker, markersize=3.5, linewidth=1.6, label=label)
            ax.grid(axis="y", alpha=.2)
            ax.set_title(model if row_idx == 0 else "")
            if col == 0:
                ax.set_ylabel("NLL" if metric == "nll" else "ECE (15 bins)")
            if row_idx == 1:
                ax.set_xlabel("S2 retention (%)")
            ax.set_xticks(x)
    axes[0, 0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "calibration_shift")


def process_differences():
    cases = [
        ("Full paired S2", "internal_test", "100_calibrated", 19),
        ("Synthetic 0% S2", "internal_test", "0_calibrated", 19),
        ("Real unqualified S2", "real_missing_s2", "all_unqualified", 15),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.1), sharex=True)
    for ax, (title, source, condition, count) in zip(axes, cases):
        if source == "internal_test":
            p = result("P", source)["conditions"][condition]
            b4 = result("B4", source)["conditions"][condition]
            differences = [p[key]["flood_iou"] - b4[key]["flood_iou"] for key in sorted(p)]
        else:
            p = result("P", source)["categories"][condition]["processes"]
            b4 = result("B4", source)["categories"][condition]["processes"]
            differences = [p[key]["scores"]["calibrated"]["flood_iou"] -
                           b4[key]["scores"]["calibrated"]["flood_iou"] for key in sorted(p)]
        if len(differences) != count or any(value is None for value in differences):
            raise RuntimeError(f"Unexpected paired process metrics: {title}")
        vals = np.asarray(differences)
        ax.axvline(0, color="#333333", linewidth=1)
        ax.scatter(vals, np.arange(count), s=19, color=COLORS["P"], alpha=.8)
        ax.axvline(vals.mean(), color=COLORS["B4"], linewidth=1.4, linestyle="--")
        ax.set(title=title, yticks=[], xlabel="P minus B4 flood IoU",
               xlim=(-.5, .2))
        ax.grid(axis="x", alpha=.18)
        ax.text(.03, .96, f"n={count}; mean={vals.mean():+.3f}",
                transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    save(fig, "process_differences")


def main():
    set_style()
    retention_curve()
    calibration_shift()
    process_differences()
    print(f"Saved three PDF and PNG figures to {OUT}")


if __name__ == "__main__":
    main()
