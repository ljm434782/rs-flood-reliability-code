"""Bounded true-class probability decomposition using the frozen B2/B4 fits.

This is a retrospective diagnostic. It reads only frozen paired role samples,
uses existing original-state temperatures, and never refits or selects a model.
Each role/model/seed/state writes an atomic, resumable job record.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))
from flood_reliability.optical_benefit import class_probability_sums
from scripts.build_upgrade_publication import DEFAULT_CONFIG, DEFAULT_SOURCE, load_complete, sha
from scripts.upgrade_calibrate_evaluate import (
    _manifest_inputs, classify_holdout_strata, open_frozen_inference_adapter,
)

OUTPUT = ROOT / "outputs" / "optical_benefit_20261001" / "class_diagnostic"
MODELS = ("B2", "B4")
ROLES = ("internal_retro", "holdout_retro")
CLASSES = ("nonwater", "permanent", "flood")
LOSSES = ("binary_flood_brier", "multiclass_brier", "multiclass_nll")


def evaluation_file(source: Path, model: str, seed: int, role: str) -> Path:
    return (source / "outputs/upgrade_20260929" / f"{model}_seed_{seed}"
            / f"evaluation_{role}_original.json")


def relevant_rows(root: Path, config: dict, role: str) -> tuple[list[dict], dict]:
    rows, owners = _manifest_inputs(root, config, role)
    if role == "internal_retro":
        return rows, owners
    eligibility_path = root / config["auxiliary_metadata"]["holdout_eligibility"]["path"]
    anchor_path = root / config["auxiliary_metadata"]["holdout_s1_anchor_audit_csv"]["path"]
    with eligibility_path.open(newline="", encoding="utf-8") as stream:
        eligibility = list(csv.DictReader(stream))
    with anchor_path.open(newline="", encoding="utf-8") as stream:
        anchors = list(csv.DictReader(stream))
    strata = classify_holdout_strata(rows, eligibility, anchors)
    return strata["paired_eligible"], owners


def expected_cell(result: dict, model: str, seed: int, role: str,
                  process: str, state: str) -> dict:
    if role == "holdout_retro":
        key = f"{model}|{seed}|original|paired_eligible|{process}|{state}|calibrated"
    else:
        key = f"{model}|{seed}|original|{process}|{state}|calibrated"
    return result["per_process"][key]


def run_job(root: Path, config: dict, source: Path, output: Path,
            model: str, seed: int, role: str, state: str, device: str) -> Path:
    job_dir = output / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    job_path = job_dir / f"{model}_{seed}_{role}_{state}.json"
    reference_path = evaluation_file(source, model, seed, role)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    run_dir = root / "outputs/upgrade_20260929" / f"{model}_seed_{seed}"
    temperature_path = run_dir / "temperature_original.json"
    checkpoint_path = run_dir / "model_only.pt"
    hashes = {"config": sha(root / "configs/upgrade_20260929.json"),
              "checkpoint": sha(checkpoint_path), "temperature": sha(temperature_path),
              "reference_evaluation": sha(reference_path),
              "manifest": config["manifests"][role]["sha256"],
              "grid": config["role_grids"][role]["sha256"]}
    if (reference["checkpoint_sha256"] != hashes["checkpoint"] or
            reference["temperature_sha256"] != hashes["temperature"] or
            reference["config_sha256"] != hashes["config"]):
        raise RuntimeError(f"staged fit/temperature differs from frozen evaluation: {model}/{seed}/{role}")
    if job_path.exists():
        existing = json.loads(job_path.read_text(encoding="utf-8"))
        if (existing.get("identity") != {"model": model, "seed": seed, "role": role,
                                          "state": state} or existing.get("hashes") != hashes or
                existing.get("status") != "COMPLETE"):
            raise RuntimeError(f"existing diagnostic job conflicts with frozen identity: {job_path}")
        return job_path
    temperature = float(json.loads(temperature_path.read_text(encoding="utf-8"))["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("frozen temperature must be positive")
    archive = "test" if role == "holdout_retro" else "train"
    adapter = open_frozen_inference_adapter(
        root, config, model, seed, root / "data/raw/impactmesh" / archive, device)
    try:
        rows, owners = relevant_rows(root, config, role)
        by_process: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            process, _ = owners[row["sample_id"]]
            by_process[process].append(row)
        if len(rows) != (432 if role == "holdout_retro" else 0) and role == "holdout_retro":
            raise RuntimeError("holdout paired sample count drifted")
        process_totals = {}
        for process in sorted(by_process):
            totals = {str(cls): {"pixels": 0, **{f"{loss}_sum": 0.0 for loss in LOSSES}}
                      for cls in range(3)}
            for row in sorted(by_process[process], key=lambda r: r["sample_id"]):
                logits, labels = adapter.predict_tile(row, state, purpose="evaluation")
                class_sums = class_probability_sums(
                    logits, labels, owners[row["sample_id"]][1], temperature=temperature)
                for cls, sums in class_sums.items():
                    for name, value in sums.items():
                        totals[str(cls)][name] += value
            pixels = sum(totals[str(cls)]["pixels"] for cls in range(3))
            expected = expected_cell(reference, model, seed, role, process, state)
            if pixels != expected["pixels"]:
                raise RuntimeError(f"class pixels differ from frozen process support: {process}")
            for loss in LOSSES:
                reconstructed = sum(totals[str(cls)][f"{loss}_sum"] for cls in range(3)) / pixels
                if not math.isclose(reconstructed, float(expected[loss]), rel_tol=0, abs_tol=5e-5):
                    raise RuntimeError(f"class {loss} does not reconstruct frozen score: {process}")
            process_totals[process] = totals
        job = {"status": "COMPLETE", "interpretation": "retrospective descriptive only",
               "identity": {"model": model, "seed": seed, "role": role, "state": state},
               "hashes": hashes, "temperature": temperature, "process_totals": process_totals,
               "process_count": len(process_totals), "tile_count": len(rows),
               "score_reconstruction_tolerance": 5e-5}
        tmp = job_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        tmp.replace(job_path)
        return job_path
    finally:
        adapter.close()


def summarize_jobs(output: Path, config: dict) -> dict:
    jobs = {}
    for model in MODELS:
        for seed in config["seeds"]:
            for role in ROLES:
                for state in (("100",) if model == "B2" else ("100", "0")):
                    path = output / "jobs" / f"{model}_{seed}_{role}_{state}.json"
                    if not path.is_file():
                        raise RuntimeError(f"class diagnostic incomplete: {path.name}")
                    job = json.loads(path.read_text(encoding="utf-8"))
                    if job["identity"] != {"model": model, "seed": seed,
                                           "role": role, "state": state}:
                        raise RuntimeError(f"class job identity mismatch: {path}")
                    jobs[model, seed, role, state] = (job, path)
    rows = []
    for (model, seed, role, state), (job, _) in jobs.items():
        for process, totals in sorted(job["process_totals"].items()):
            total_pixels = sum(totals[str(cls)]["pixels"] for cls in range(3))
            for cls, label in enumerate(CLASSES):
                item = totals[str(cls)]
                row = {"model": model, "seed": seed, "role": role,
                       "stratum": "paired_eligible" if role == "holdout_retro" else "paired",
                       "state": state, "process_id": process, "true_class": label,
                       "class_pixels": item["pixels"], "total_pixels": total_pixels,
                       "class_prevalence": item["pixels"] / total_pixels}
                for loss in LOSSES:
                    total = item[f"{loss}_sum"]
                    row[f"{loss}_sum"] = total
                    row[f"{loss}_conditional_mean"] = (
                        total / item["pixels"] if item["pixels"] else None)
                    row[f"{loss}_all_pixel_contribution"] = total / total_pixels
                rows.append(row)
    path = output / "class_process.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {"status": "COMPLETE", "interpretation": "retrospective descriptive only",
                  "scope": "B2 once per role/seed (architecture is S2-invariant); B4 at 100/0",
                  "models": MODELS, "seeds": config["seeds"], "roles": ROLES,
                  "job_hashes": {path.name: sha(path) for _, path in jobs.values()},
                  "class_process_sha256": sha(path), "script_sha256": sha(Path(__file__))}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n",
                                            encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="isolated, hash-matched package with original raw-data link and staged B2/B4 fits")
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    parser.add_argument("--max-new-jobs", type=int, default=100)
    args = parser.parse_args()
    root = args.root.resolve()
    config, audit, _, _ = load_complete(DEFAULT_SOURCE, DEFAULT_CONFIG)
    if audit["evaluation_record_count"] != {"internal_retro": 24, "holdout_retro": 24}:
        raise RuntimeError("full frozen matrix required")
    if sha(root / "configs/upgrade_20260929.json") != sha(DEFAULT_CONFIG):
        raise RuntimeError("staged config differs from published frozen config")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    new_jobs = 0
    for model in MODELS:
        for seed in config["seeds"]:
            for role in ROLES:
                for state in (("100",) if model == "B2" else ("100", "0")):
                    path = OUTPUT / "jobs" / f"{model}_{seed}_{role}_{state}.json"
                    was_missing = not path.exists()
                    if was_missing and new_jobs >= args.max_new_jobs:
                        print(f"PAUSED after {new_jobs} new jobs")
                        return
                    run_job(root, config, DEFAULT_SOURCE, OUTPUT,
                            model, seed, role, state, args.device)
                    if path.exists():
                        print(f"VERIFIED {path.name}", flush=True)
                    if was_missing:
                        new_jobs += 1
    result = summarize_jobs(OUTPUT, config)
    print(json.dumps({"status": result["status"], "jobs": len(result["job_hashes"])}))


if __name__ == "__main__":
    main()
