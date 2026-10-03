"""Rebuild the manuscript's two main tables and figures from frozen CSVs.

This deliberately reads no raster, checkpoint, label, or evaluation JSON. It
reproduces presentation artifacts from the archived retrospective summaries,
not the underlying inference or a new validation experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROLES = (("internal_retro", "Internal", 18), ("holdout_retro", "Holdout", 9))
STATES = ("100", "75", "50", "25", "0")
CLASSES = ("nonwater", "permanent", "flood")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite frozen metric")
    return number


def format_benefit_table(rows: list[dict[str, str]]) -> str:
    cells = {(row["role"], row["state"]): row for row in rows
             if row["comparator"] == "B4"}
    lines = [r"\begin{table}[H]",
             r"\caption{Retrospective B4 benefit relative to the frozen S1-only B2 on the same owned pixels. Positive values favor B4. Entries average process-level gains over three fixed seeds; 95\% descriptive process intervals are in the accompanying CSV. Outcome counts classify process-level three-seed means. These are synthetic optical-retention conditions, not natural absence.}",
             r"\label{tab:opticalbenefit}", r"\centering\small",
             r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.15}",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\hline",
             r"Role & S2 kept & Flood IoU gain & Flood Brier gain & Both better & Both worse \\",
             r"\hline"]
    for role, label, expected_n in ROLES:
        for state in STATES:
            row = cells[role, state]
            n = int(row["process_n_joint"])
            if n != expected_n:
                raise ValueError(f"unexpected process support for {role}/{state}: {n}")
            iou = _finite(row["flood_iou_gain_mean"])
            brier = _finite(row["binary_flood_brier_gain_mean"])
            lines.append(f"{label} & {state}\\% & {iou:+.4f} & {brier:+.4f} & "
                         f"{row['outcome_both_better']}/{n} & "
                         f"{row['outcome_both_worse']}/{n}" + r" \\")
        if role == "internal_retro":
            lines.append(r"\hline")
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines) + "\n"


def format_class_table(rows: list[dict[str, str]],
                       benefit_rows: list[dict[str, str]]) -> str:
    cells = {(row["role"], row["state"], row["true_class"]): row for row in rows}
    gains = {(row["role"], row["state"]): row for row in benefit_rows
             if row["comparator"] == "B4"}
    lines = [r"\begin{table}[H]",
             r"\caption{True-class contributions to B2-minus-B4 flood Brier on paired process pixels. Each entry is a process-equal three-seed mean contribution to the all-pixel score; the three classes sum to the total gain. Positive favors B4. Class-conditional means, class support, and descriptive intervals are archived separately.}",
             r"\label{tab:opticalclass}", r"\centering\small",
             r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.15}",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\hline",
             r"Role & S2 kept & Non-water & Permanent & Flood & Total \\", r"\hline"]
    for role, label, _ in ROLES:
        for state in ("100", "0"):
            values = [_finite(cells[role, state, cls]["binary_flood_brier_all_pixel_gain_mean"])
                      for cls in CLASSES]
            total = sum(values)
            expected = _finite(gains[role, state]["binary_flood_brier_gain_mean"])
            if not math.isclose(total, expected, rel_tol=0, abs_tol=5e-5):
                raise ValueError(f"class contributions do not reconstruct Brier gain: {role}/{state}")
            lines.append(f"{label} & {state}\\% & " + " & ".join(f"{v:+.4f}" for v in values)
                         + f" & {total:+.4f}" + r" \\")
        if role == "internal_retro":
            lines.append(r"\hline")
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines) + "\n"


def _check_provenance(base: Path) -> None:
    primary = json.loads((base / "provenance.json").read_text(encoding="utf-8"))
    class_record = json.loads((base / "class_gain_provenance.json").read_text(encoding="utf-8"))
    if (primary.get("status") != "COMPLETE" or
            primary.get("interpretation") != "retrospective descriptive only" or
            primary.get("seeds") != [20261013, 20261014, 20261015] or
            primary.get("states") != list(STATES) or
            class_record.get("status") != "COMPLETE"):
        raise ValueError("frozen retrospective summary identity is incomplete")
    for name in ("summary.csv", "process_mean_gain.csv"):
        if _sha(base / name) != primary["generated_hashes"][name]:
            raise ValueError(f"frozen source hash mismatch: {name}")
    if (_sha(base / "class_gain_summary.csv") !=
            class_record["generated_hashes"]["class_gain_summary.csv"] or
            _sha(base / "summary.csv") != class_record["source_benefit_summary_sha256"]):
        raise ValueError("class contribution provenance mismatch")


def render_cached(root: Path, output: Path) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = root / "outputs/optical_benefit_20261001"
    _check_provenance(base)
    summary = _rows(base / "summary.csv")
    classes = _rows(base / "class_gain_summary.csv")
    output.mkdir(parents=True, exist_ok=True)
    (output / "optical_benefit_table.tex").write_text(format_benefit_table(summary), encoding="utf-8")
    (output / "optical_class_contributions.tex").write_text(
        format_class_table(classes, summary), encoding="utf-8")

    import sys
    sys.path.insert(0, str(root))
    from scripts.plot_rs_optical_benefit_origin import draw_origin_benefit
    draw_origin_benefit(summary, output / "optical_benefit_joint.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.5), sharey=True)
    class_colors = {"nonwater": "#7e9ea9", "permanent": "#ad813c", "flood": "#3b7d7a"}
    for ax, (role, _, _) in zip(axes, ROLES):
        for i, state in enumerate(("100", "0")):
            for cls in CLASSES:
                row = next(r for r in classes if r["role"] == role and
                           r["state"] == state and r["true_class"] == cls)
                ax.bar(i + (CLASSES.index(cls) - 1) * .22,
                       _finite(row["binary_flood_brier_all_pixel_gain_mean"]),
                       width=.2, color=class_colors[cls], label=cls if i == 0 else None)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xticks((0, 1), ("100%", "0%"))
        ax.set(xlabel="S2 retained", title=role.replace("_", " "))
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Contribution to flood Brier gain vs S1-only")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "optical_class_contributions.pdf")
    plt.close(fig)

    names = ("optical_benefit_table.tex", "optical_class_contributions.tex",
             "optical_benefit_joint.pdf", "optical_class_contributions.pdf")
    return {name: _sha(output / name) for name in names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.root / "regenerated_main_artifacts"
    print(json.dumps(render_cached(args.root, destination), indent=2))
