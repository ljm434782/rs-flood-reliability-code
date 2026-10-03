"""Build fixed B2-relative optical-benefit diagnostics from the complete matrix.

No raster, checkpoint, or label is opened here. All outputs are retrospective.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path


import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from flood_reliability.optical_benefit import classify_joint_gain, pair_process_cells
from scripts.build_upgrade_publication import DEFAULT_CONFIG, DEFAULT_SOURCE, load_complete, sha

OUTPUT = ROOT / "outputs" / "optical_benefit_20261001"
STATES = ("100", "75", "50", "25", "0")
COMPARATORS = (("B4", "B4"), ("B5_whole", "A"), ("B5_local", "C"),
               ("P", "P"), ("SMAGNet", "SMAGNet"))
ROLE_STRATA = (("internal_retro", "paired"), ("holdout_retro", "paired_eligible"))
OUTCOMES = ("both_better", "segmentation_only", "probability_only", "both_worse",
            "tie", "undefined_iou")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("diagnostic table cannot be empty")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluation(source: Path, model: str, seed: int, role: str) -> tuple[dict, Path]:
    path = (source / "outputs/upgrade_20260929" / f"{model}_seed_{seed}"
            / f"evaluation_{role}_original.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result["model"], result["seed"], result["role"],
            result["calibration_strategy"]) != (model, seed, role, "original"):
        raise ValueError(f"evaluation identity mismatch: {path}")
    return result, path


def cell(result: dict, process: str, state: str, stratum: str) -> dict:
    prefix = f"{result['model']}|{result['seed']}|original|"
    suffix = (f"{stratum}|{process}|{state}|calibrated" if result["role"] == "holdout_retro"
              else f"{process}|{state}|calibrated")
    return result["per_process"][prefix + suffix]


def process_ids(result: dict, role: str, stratum: str) -> tuple[str, ...]:
    found = set()
    for key in result["per_process"]:
        bits = key.split("|")
        if bits[-2:] != ["100", "calibrated"]:
            continue
        if role == "holdout_retro":
            if bits[-4] != stratum:
                continue
            found.add(bits[-3])
        else:
            found.add(bits[-3])
    if not found:
        raise ValueError(f"no paired processes for {role}/{stratum}")
    return tuple(sorted(found))


def summarize(rows: list[dict], seeds: tuple[int, ...], draws: int,
              rng: np.random.Generator) -> tuple[dict, list[dict], dict]:
    processes = sorted({row["process_id"] for row in rows})
    lookup = {(row["process_id"], row["seed"]): row for row in rows}
    if len(lookup) != len(processes) * len(seeds) or len(rows) != len(lookup):
        raise ValueError("paired process/seed matrix is incomplete or duplicated")
    process_rows = []
    for process in processes:
        members = [lookup[process, seed] for seed in seeds]
        ious = [r["flood_iou_gain"] for r in members]
        iou = float(np.mean(ious)) if all(value is not None for value in ious) else None
        brier = float(np.mean([r["binary_flood_brier_gain"] for r in members]))
        process_rows.append({"process_id": process, "flood_iou_gain": iou,
                             "binary_flood_brier_gain": brier,
                             "joint_outcome": classify_joint_gain(iou, brier)})
    counts = Counter(r["joint_outcome"] for r in process_rows)
    brier = np.array([r["binary_flood_brier_gain"] for r in process_rows])
    valid_iou = [r["flood_iou_gain"] for r in process_rows if r["flood_iou_gain"] is not None]
    selections = rng.integers(0, len(processes), size=(draws, len(processes)))
    brier_ci = np.quantile(brier[selections].mean(axis=1), (0.025, 0.975))
    if valid_iou:
        iou_values = np.asarray(valid_iou)
        selected_iou = rng.integers(0, len(valid_iou), size=(draws, len(valid_iou)))
        iou_ci = np.quantile(iou_values[selected_iou].mean(axis=1), (0.025, 0.975))
    else:
        iou_ci = (None, None)
    summary = {"process_n_brier": len(processes), "process_n_joint": len(valid_iou),
               "flood_iou_gain_mean": float(np.mean(valid_iou)) if valid_iou else None,
               "flood_iou_gain_ci95_low": iou_ci[0], "flood_iou_gain_ci95_high": iou_ci[1],
               "binary_flood_brier_gain_mean": float(brier.mean()),
               "binary_flood_brier_gain_ci95_low": float(brier_ci[0]),
               "binary_flood_brier_gain_ci95_high": float(brier_ci[1]),
               **{f"outcome_{outcome}": counts[outcome] for outcome in OUTCOMES}}
    return summary, process_rows, counts


def plot_b4(summary_rows: list[dict], output: Path) -> None:
    from scripts.plot_rs_optical_benefit_origin import draw_origin_benefit
    draw_origin_benefit(summary_rows, output / "b4_joint_optical_benefit.pdf",
                        output / "b4_joint_optical_benefit.png")


def render_latex_table(summaries: list[dict], target: Path) -> None:
    cells = {(r["role"], r["state"]): r for r in summaries if r["comparator"] == "B4"}
    lines = [r"\begin{table}[H]", r"\caption{Retrospective B4 benefit relative to the frozen S1-only B2 on the same owned pixels. Positive values favor B4. Entries average process-level gains over three fixed seeds; 95\% descriptive process intervals are in the accompanying CSV. Outcome counts classify process-level three-seed means. These are synthetic optical-retention conditions, not natural absence.}",
             r"\label{tab:opticalbenefit}", r"\centering\small",
             r"\setlength{\tabcolsep}{5pt}", r"\renewcommand{\arraystretch}{1.15}",
             r"\begin{tabular}{@{}llrrrr@{}}", r"\hline",
             r"Role & S2 kept & Flood IoU gain & Flood Brier gain & Both better & Both worse \\",
             r"\hline"]
    for role, label in (("internal_retro", "Internal"), ("holdout_retro", "Holdout")):
        for state in STATES:
            row = cells[role, state]
            iou = row["flood_iou_gain_mean"]
            brier = row["binary_flood_brier_gain_mean"]
            lines.append(f"{label} & {state}\\% & {iou:+.4f} & {brier:+.4f} & "
                         f"{row['outcome_both_better']}/{row['process_n_joint']} & "
                         f"{row['outcome_both_worse']}/{row['process_n_joint']} \\\\")
        if role == "internal_retro":
            lines.append(r"\hline")
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}"))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(source: Path = DEFAULT_SOURCE, config_path: Path = DEFAULT_CONFIG,
           output: Path = OUTPUT) -> dict:
    config, audit, _, _ = load_complete(source, config_path)
    if audit["evaluation_record_count"] != {"internal_retro": 24, "holdout_retro": 24}:
        raise ValueError("complete evaluation matrix required")
    seeds = tuple(config["seeds"])
    bootstrap = config["evaluation_protocol"]["bootstrap"]
    output.mkdir(parents=True, exist_ok=True)
    all_rows, mean_rows, summaries, source_hashes = [], [], [], {}
    natural_rows = []
    for role, stratum in ROLE_STRATA:
        for seed in seeds:
            baseline, base_path = evaluation(source, "B2", seed, role)
            source_hashes[base_path.relative_to(source).as_posix()] = sha(base_path)
            processes = process_ids(baseline, role, stratum)
            for model, display in COMPARATORS:
                compared, path = evaluation(source, model, seed, role)
                source_hashes[path.relative_to(source).as_posix()] = sha(path)
                if process_ids(compared, role, stratum) != processes:
                    raise ValueError(f"unpaired process IDs: {role}/{model}/{seed}")
                for state in STATES:
                    for process in processes:
                        pair = pair_process_cells(cell(baseline, process, state, stratum),
                                                  cell(compared, process, state, stratum))
                        all_rows.append({"role": role, "stratum": stratum,
                                         "comparator": display, "model": model,
                                         "seed": seed, "process_id": process,
                                         "state": state, **pair})
                if role == "holdout_retro" and model == "B4":
                    for natural_stratum, natural_state in (("low_visible", "observed"),
                                                            ("no_qualifying_s2", "observed_absent")):
                        for process in process_ids_natural(baseline, natural_stratum, natural_state):
                            pair = pair_process_cells(
                                cell(baseline, process, natural_state, natural_stratum),
                                cell(compared, process, natural_state, natural_stratum))
                            natural_rows.append({"role": role, "stratum": natural_stratum,
                                                 "comparator": "B4", "seed": seed,
                                                 "process_id": process, "state": natural_state, **pair})
    rng = np.random.Generator(np.random.PCG64(bootstrap["seed"]))
    for role, stratum in ROLE_STRATA:
        for model, display in COMPARATORS:
            for state in STATES:
                subset = [r for r in all_rows if r["role"] == role and
                          r["comparator"] == display and r["state"] == state]
                summary, process_means, _ = summarize(subset, seeds,
                                                       bootstrap["replicates"], rng)
                summaries.append({"role": role, "stratum": stratum,
                                  "comparator": display, "state": state, **summary})
                mean_rows.extend({"role": role, "stratum": stratum,
                                  "comparator": display, "state": state, **row}
                                 for row in process_means)
    write_csv(output / "process_seed_gain.csv", all_rows)
    write_csv(output / "process_mean_gain.csv", mean_rows)
    write_csv(output / "summary.csv", summaries)
    unfavorable = [row for row in mean_rows if row["comparator"] == "B4"
                   and row["state"] in ("25", "0") and row["joint_outcome"] == "both_worse"]
    write_csv(output / "unfavorable_processes.csv", unfavorable)
    write_csv(output / "natural_holdout_gain.csv", natural_rows)
    natural_summary = []
    for stratum, state in (("low_visible", "observed"),
                           ("no_qualifying_s2", "observed_absent")):
        subset = [r for r in natural_rows if r["stratum"] == stratum]
        summary, _, _ = summarize(subset, seeds, bootstrap["replicates"], rng)
        natural_summary.append({"role": "holdout_retro", "stratum": stratum,
                                "comparator": "B4", "state": state, **summary})
    write_csv(output / "natural_holdout_summary.csv", natural_summary)
    plot_b4(summaries, output)
    paper_table = ROOT / "paper/tables/optical_benefit_table.tex"
    paper_figure = ROOT / "paper/figures/optical_benefit_joint.pdf"
    render_latex_table(summaries, paper_table)
    shutil.copyfile(output / "b4_joint_optical_benefit.pdf", paper_figure)
    generated = [output / name for name in ("process_seed_gain.csv", "process_mean_gain.csv",
                 "summary.csv", "unfavorable_processes.csv", "natural_holdout_gain.csv",
                 "natural_holdout_summary.csv",
                 "b4_joint_optical_benefit.pdf", "b4_joint_optical_benefit.png")]
    provenance = {"status": "COMPLETE", "interpretation": "retrospective descriptive only",
                  "models": [item[0] for item in COMPARATORS], "baseline": "B2",
                  "seeds": seeds, "states": STATES,
                  "temperature_strategy": "original; one fixed positive temperature per model and seed",
                  "score_direction": "positive gain means fusion is better than B2",
                  "missing_iou": "undefined remains undefined; common three-seed process support for joint outcomes",
                  "roles_separate": True, "natural_no_s2_separate": True,
                  "bootstrap": bootstrap, "matrix_audit_sha256": sha(
                      source / "outputs/upgrade_20260929/summaries/matrix_summary_audit.json"),
                  "config_sha256": sha(config_path), "source_evaluation_hashes": source_hashes,
                  "script_sha256": sha(Path(__file__)),
                  "plot_script_sha256": sha(ROOT / "scripts/plot_rs_optical_benefit_origin.py"),
                  "generated_hashes": {p.name: sha(p) for p in generated},
                  "paper_artifact_hashes": {p.relative_to(ROOT).as_posix(): sha(p)
                                            for p in (paper_table, paper_figure)}}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                            encoding="utf-8")
    return provenance


def process_ids_natural(result: dict, stratum: str, state: str) -> tuple[str, ...]:
    found = set()
    for key in result["per_process"]:
        bits = key.split("|")
        if len(bits) == 7 and bits[-4] == stratum and bits[-2:] == [state, "calibrated"]:
            found.add(bits[-3])
    if not found:
        raise ValueError(f"natural holdout stratum has no processes: {stratum}")
    return tuple(sorted(found))


if __name__ == "__main__":
    result = render()
    print(json.dumps({"status": result["status"], "source_files":
                      len(result["source_evaluation_hashes"])}))
