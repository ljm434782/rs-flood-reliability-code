"""Summarize frozen B2/B4 true-class probability contributions, without inference."""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from flood_reliability.optical_benefit import pair_class_rows
from scripts.build_upgrade_publication import DEFAULT_CONFIG, sha

OUTPUT = ROOT / "outputs/optical_benefit_20261001"
CLASSES = ("nonwater", "permanent", "flood")
ROLES = ("internal_retro", "holdout_retro")
STATES = ("100", "0")
LOSSES = ("binary_flood_brier", "multiclass_brier", "multiclass_nll")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty class diagnostic")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render() -> dict:
    source = OUTPUT / "class_diagnostic/class_process.csv"
    provenance_path = OUTPUT / "class_diagnostic/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "COMPLETE" or provenance.get("class_process_sha256") != sha(source):
        raise RuntimeError("class inference must be complete and hash-matched")
    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    seeds = tuple(config["seeds"])
    all_rows = read_csv(source)
    lookup = {(r["model"], int(r["seed"]), r["role"], r["state"],
               r["process_id"], r["true_class"]): r for r in all_rows}
    if len(lookup) != len(all_rows):
        raise ValueError("duplicate class inference row")
    gains = []
    for role in ROLES:
        for seed in seeds:
            processes = sorted({key[4] for key in lookup if key[:4] == ("B2", seed, role, "100")})
            if len(processes) != (18 if role == "internal_retro" else 9):
                raise ValueError("unexpected paired process count")
            for process in processes:
                for state in STATES:
                    for cls in CLASSES:
                        sar = lookup["B2", seed, role, "100", process, cls]
                        fusion = lookup["B4", seed, role, state, process, cls]
                        sar_values = {name: float(value) if name.endswith("_sum") else int(value)
                                      for name, value in sar.items()
                                      if name in ("class_pixels", "total_pixels") or name.endswith("_sum")}
                        fusion_values = {name: float(value) if name.endswith("_sum") else int(value)
                                         for name, value in fusion.items()
                                         if name in ("class_pixels", "total_pixels") or name.endswith("_sum")}
                        paired = pair_class_rows(sar_values, fusion_values)
                        gains.append({"role": role, "seed": seed, "state": state,
                                      "process_id": process, "true_class": cls, **paired})
    write_csv(OUTPUT / "class_gain_process_seed.csv", gains)
    summary = []
    rng = np.random.Generator(np.random.PCG64(config["evaluation_protocol"]["bootstrap"]["seed"]))
    draws = config["evaluation_protocol"]["bootstrap"]["replicates"]
    for role in ROLES:
        for state in STATES:
            for cls in CLASSES:
                group = [r for r in gains if r["role"] == role and r["state"] == state
                         and r["true_class"] == cls]
                processes = sorted({r["process_id"] for r in group})
                if len(group) != len(processes) * len(seeds):
                    raise ValueError("incomplete class/seed support")
                by_key = {(r["process_id"], r["seed"]): r for r in group}
                row = {"role": role, "state": state, "true_class": cls,
                       "process_n_all_pixel": len(processes),
                       "process_n_class_present": sum(
                           all(by_key[p, seed]["class_pixels"] > 0 for seed in seeds)
                           for p in processes)}
                for loss in LOSSES:
                    values = np.asarray([
                        np.mean([by_key[p, seed][f"{loss}_all_pixel_gain"] for seed in seeds])
                        for p in processes])
                    index = rng.integers(0, len(values), size=(draws, len(values)))
                    ci = np.quantile(values[index].mean(axis=1), (0.025, 0.975))
                    row[f"{loss}_all_pixel_gain_mean"] = float(values.mean())
                    row[f"{loss}_all_pixel_gain_ci95_low"] = float(ci[0])
                    row[f"{loss}_all_pixel_gain_ci95_high"] = float(ci[1])
                    conditional = [np.mean([
                        by_key[p, seed][f"{loss}_conditional_gain"] for seed in seeds])
                        for p in processes if all(by_key[p, seed]["class_pixels"] > 0 for seed in seeds)]
                    row[f"{loss}_conditional_gain_mean"] = (
                        float(np.mean(conditional)) if conditional else None)
                summary.append(row)
    write_csv(OUTPUT / "class_gain_summary.csv", summary)
    benefit = {(r["role"], r["state"]): r for r in read_csv(OUTPUT / "summary.csv")
               if r["comparator"] == "B4" and r["state"] in STATES}
    for role in ROLES:
        for state in STATES:
            subtotal = sum(r["binary_flood_brier_all_pixel_gain_mean"] for r in summary
                           if r["role"] == role and r["state"] == state)
            full = float(benefit[role, state]["binary_flood_brier_gain_mean"])
            if not math.isclose(subtotal, full, rel_tol=0, abs_tol=5e-5):
                raise RuntimeError(f"true-class contribution does not reconstruct paired benefit: {role}/{state}")
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.5), sharey=True)
    colors = {"nonwater": "#7e9ea9", "permanent": "#ad813c", "flood": "#3b7d7a"}
    for ax, role in zip(axes, ROLES):
        for i, state in enumerate(STATES):
            rows = [r for r in summary if r["role"] == role and r["state"] == state]
            for cls in CLASSES:
                value = next(r["binary_flood_brier_all_pixel_gain_mean"] for r in rows
                             if r["true_class"] == cls)
                ax.bar(i + (CLASSES.index(cls) - 1) * .22, value, width=.2,
                       color=colors[cls], label=cls if i == 0 else None)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_xticks((0, 1), ("100%", "0%"))
        ax.set(xlabel="S2 retained", title=role.replace("_", " "))
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Contribution to flood Brier gain vs S1-only")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT / f"class_brier_contributions.{ext}", dpi=180)
    plt.close(fig)
    lines = [r"\begin{table}[H]", r"\caption{True-class contributions to B2-minus-B4 flood Brier on paired process pixels. Each entry is a process-equal three-seed mean contribution to the all-pixel score; the three classes sum to the total gain. Positive favors B4. Class-conditional means, class support, and descriptive intervals are archived separately.}",
             r"\label{tab:opticalclass}", r"\centering\small",
             r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.15}",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\hline",
             r"Role & S2 kept & Non-water & Permanent & Flood & Total \\", r"\hline"]
    for role, label in (("internal_retro", "Internal"), ("holdout_retro", "Holdout")):
        for state in STATES:
            values = [next(r["binary_flood_brier_all_pixel_gain_mean"] for r in summary
                           if r["role"] == role and r["state"] == state and r["true_class"] == cls)
                      for cls in CLASSES]
            lines.append(f"{label} & {state}\\% & " + " & ".join(f"{v:+.4f}" for v in values)
                         + f" & {sum(values):+.4f} \\\\")
        if role == "internal_retro":
            lines.append(r"\hline")
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}"))
    table = ROOT / "paper/tables/optical_class_contributions.tex"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure = ROOT / "paper/figures/optical_class_contributions.pdf"
    shutil.copyfile(OUTPUT / "class_brier_contributions.pdf", figure)
    generated = [OUTPUT / name for name in ("class_gain_process_seed.csv", "class_gain_summary.csv",
                 "class_brier_contributions.pdf", "class_brier_contributions.png")]
    record = {"status": "COMPLETE", "interpretation": "retrospective descriptive only",
              "source_class_process_sha256": sha(source), "source_provenance_sha256": sha(provenance_path),
              "source_benefit_summary_sha256": sha(OUTPUT / "summary.csv"),
              "script_sha256": sha(Path(__file__)),
              "generated_hashes": {p.name: sha(p) for p in generated},
              "paper_artifact_hashes": {p.relative_to(ROOT).as_posix(): sha(p)
                                        for p in (table, figure)}}
    (OUTPUT / "class_gain_provenance.json").write_text(json.dumps(record, indent=2) + "\n",
                                                         encoding="utf-8")
    return record


if __name__ == "__main__":
    result = render()
    print(json.dumps({"status": result["status"], "outputs": len(result["generated_hashes"])}))
