"""Render class IoU and flood reliability diagnostics from the complete matrix.

All units for intervals are retrospective flood processes. No raster or label
pixels are reopened, and this script does not select models or thresholds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.build_upgrade_publication import DEFAULT_CONFIG, DEFAULT_OUTPUT, DEFAULT_SOURCE, load_complete


STATES = ("100", "75", "50", "25", "0", "irregular_75", "irregular_50", "irregular_25")
CLASS_NAMES = ("nonwater", "permanent", "flood")
ARMS = {"A": ("B5_whole", "original"), "B": ("B5_whole", "multistate"),
        "C": ("B5_local", "original"), "D": ("B5_local", "multistate")}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No diagnostic rows")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def interval(values: np.ndarray, draws: np.ndarray) -> tuple[float | None, float | None]:
    """Resample process rows jointly across fixed seed columns."""
    samples = values[draws]
    counts = np.isfinite(samples).sum(axis=1)
    if np.any(counts == 0):
        return None, None
    seed_means = np.nansum(samples, axis=1) / counts
    scores = seed_means.mean(axis=1)
    lo, hi = np.quantile(scores, (0.025, 0.975))
    return float(lo), float(hi)


def render(source: Path = DEFAULT_SOURCE, config_path: Path = DEFAULT_CONFIG,
           output: Path = DEFAULT_OUTPUT) -> dict:
    config, audit, _, _ = load_complete(source, config_path)
    if audit["evaluation_record_count"] != {"internal_retro": 24, "holdout_retro": 24}:
        raise ValueError("Complete frozen evaluation matrix is required")
    base = source / "outputs/upgrade_20260929"
    seeds = tuple(int(x) for x in config["seeds"])
    processes = tuple(audit["primary_interaction"]["expected_process_ids"])
    if len(processes) != 18:
        raise ValueError("Unexpected retrospective internal process support")
    protocol = config["evaluation_protocol"]["bootstrap"]
    draws = np.random.Generator(np.random.PCG64(protocol["seed"])).integers(
        0, len(processes), size=(protocol["replicates"], len(processes)))
    class_rows: list[dict] = []
    class_groups: dict[tuple[str, str, str, str], np.ndarray] = {}
    reliability_groups: dict[tuple[str, str, str, int], dict[str, list[float]]] = {}
    source_hashes = {}
    for model, spec in config["models"].items():
        for seed_index, seed in enumerate(seeds):
            for strategy in spec["calibration"]:
                path = base / f"{model}_seed_{seed}" / f"evaluation_internal_retro_{strategy}.json"
                result = json.loads(path.read_text(encoding="utf-8"))
                if (result["model"] != model or result["seed"] != seed
                        or result["calibration_strategy"] != strategy
                        or result["role"] != "internal_retro"):
                    raise ValueError("Evaluation identity mismatch: " + str(path))
                source_hashes[path.relative_to(source).as_posix()] = digest(path)
                for process_index, process in enumerate(processes):
                    for state in STATES:
                        for score in ("raw", "calibrated"):
                            key = f"{model}|{seed}|{strategy}|{process}|{state}|{score}"
                            cell = result["per_process"].get(key)
                            if cell is None:
                                raise ValueError("Missing complete process/state cell: " + key)
                            if score == "calibrated":
                                for cls_index, cls in enumerate(CLASS_NAMES):
                                    value = cell["class_iou"][cls_index]
                                    row = {"model": model, "seed": seed,
                                           "strategy": strategy, "process_id": process,
                                           "state": state, "class": cls,
                                           "class_iou": value, "pixels": cell["pixels"]}
                                    class_rows.append(row)
                                    group = (model, strategy, state, cls)
                                    if group not in class_groups:
                                        class_groups[group] = np.full((len(processes), len(seeds)), np.nan)
                                    if value is not None:
                                        class_groups[group][process_index, seed_index] = value
                            if (model, strategy) not in ARMS.values() or state not in ("100", "0"):
                                continue
                            arm = next(name for name, identity in ARMS.items()
                                       if identity == (model, strategy))
                            reliability = cell["reliability"]
                            counts = reliability["count"]
                            predicted = reliability["confidence_sum"]
                            observed = reliability["observed_rate_sum"]
                            if not (len(counts) == len(predicted) == len(observed) == 15):
                                raise ValueError("Frozen reliability bins are not 15")
                            for bin_index in range(15):
                                item = reliability_groups.setdefault(
                                    (arm, state, score, bin_index),
                                    {p: [0.0, 0.0, 0.0] for p in processes})
                                item[process][0] += counts[bin_index]
                                item[process][1] += predicted[bin_index]
                                item[process][2] += observed[bin_index]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "classwise_process.csv", class_rows)
    class_summary = []
    for (model, strategy, state, cls), values in sorted(class_groups.items()):
        counts = np.isfinite(values).sum(axis=0)
        per_seed = np.nansum(values, axis=0) / np.maximum(counts, 1)
        mean = float(per_seed.mean()) if np.all(counts) else None
        lo, hi = interval(values, draws) if mean is not None else (None, None)
        class_summary.append({"model": model, "strategy": strategy,
                              "arm": next((arm for arm, identity in ARMS.items()
                                           if identity == (model, strategy)), ""),
                              "state": state, "class": cls, "mean": mean,
                              "ci95_low": lo, "ci95_high": hi,
                              "process_n_min": int(counts.min()),
                              **{f"seed_{seed}": float(per_seed[i]) if counts[i] else None
                                 for i, seed in enumerate(seeds)}})
    write_csv(output / "classwise_summary.csv", class_summary)
    reliability_rows = []
    for (arm, state, score, bin_index), item in sorted(reliability_groups.items()):
        counts = np.array([item[p][0] for p in processes])
        pred = np.array([item[p][1] for p in processes])
        obs = np.array([item[p][2] for p in processes])
        support = counts > 0
        pred_rate = np.divide(pred, counts, out=np.full_like(pred, np.nan), where=support)
        obs_rate = np.divide(obs, counts, out=np.full_like(obs, np.nan), where=support)
        # Each process pools the three fixed seeds within a bin, then processes
        # receive equal weight. Empty-bin processes remain explicit undefined.
        mean_pred = float(np.nanmean(pred_rate)) if support.any() else None
        mean_obs = float(np.nanmean(obs_rate)) if support.any() else None
        lo, hi = interval(obs_rate[:, None], draws) if support.any() else (None, None)
        reliability_rows.append({"arm": arm, "model": ARMS[arm][0],
                                 "strategy": ARMS[arm][1], "state": state,
                                 "score": score, "bin": bin_index,
                                 "bin_left": bin_index / 15,
                                 "bin_right": (bin_index + 1) / 15,
                                 "process_n": int(support.sum()),
                                 "pixel_count_across_fixed_seeds": int(counts.sum()),
                                 "mean_predicted_flood_probability": mean_pred,
                                 "observed_flood_frequency": mean_obs,
                                 "observed_ci95_low": lo, "observed_ci95_high": hi})
    write_csv(output / "flood_reliability_bins.csv", reliability_rows)
    for state in ("100", "0"):
        fig, ax = plt.subplots(figsize=(6.2, 5.0))
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1,
                label="Ideal")
        for arm in "ABCD":
            selected = [r for r in reliability_rows
                        if r["arm"] == arm and r["state"] == state
                        and r["score"] == "calibrated"
                        and r["process_n"] > 0]
            x = [r["mean_predicted_flood_probability"] for r in selected]
            y = [r["observed_flood_frequency"] for r in selected]
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.2, label=arm)
            low = [r["observed_ci95_low"] if r["observed_ci95_low"] is not None else math.nan
                   for r in selected]
            high = [r["observed_ci95_high"] if r["observed_ci95_high"] is not None else math.nan
                    for r in selected]
            ax.fill_between(x, low, high, alpha=0.09)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted flood probability",
               ylabel="Observed flood frequency",
               title=f"Internal retrospective reliability, {state}% optical blocks retained")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=5, loc="upper left")
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(output / f"flood_reliability_{state}.{suffix}", dpi=180)
        plt.close(fig)
    generated = [output / name for name in (
        "classwise_process.csv", "classwise_summary.csv", "flood_reliability_bins.csv",
        "flood_reliability_100.pdf", "flood_reliability_100.png",
        "flood_reliability_0.pdf", "flood_reliability_0.png")]
    provenance = {"status": "COMPLETE", "role": "internal_retro",
                  "interpretation": "retrospective descriptive only",
                  "process_unit": "conservative flood process",
                  "seed_rule": "pool fixed seeds within process for reliability; keep seeds joint for IoU bootstrap",
                  "bootstrap": protocol, "process_ids": processes,
                  "matrix_audit_sha256": digest(base / "summaries/matrix_summary_audit.json"),
                  "source_evaluation_hashes": source_hashes,
                  "generator_sha256": digest(Path(__file__)),
                  "generated_file_hashes": {p.name: digest(p) for p in generated}}
    (output / "diagnostics_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


if __name__ == "__main__":
    record = render()
    print(json.dumps({"status": record["status"],
                      "outputs": len(record["generated_file_hashes"])}))
