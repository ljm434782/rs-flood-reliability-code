"""Evaluate the already-fitted B1 hard SAR baseline on frozen retrospective roles.

This path reports hard-label segmentation metrics only. It does not fit a
threshold, temperature, or any parameter and never contributes to the B5
primary probability estimand.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from flood_reliability.impactmesh_io import TrainArchiveReader
from scripts.fit_impactmesh_b1 import predict, pre_event_vh
from scripts.upgrade_calibrate_evaluate import (
    classify_holdout_strata, manifest_owner_masks, require_real_access, sha256,
)


def _hard_metrics(confusion: np.ndarray) -> dict:
    matrix = np.asarray(confusion, dtype=np.int64)
    rows = []
    for cls in range(3):
        tp = int(matrix[cls, cls])
        fp = int(matrix[:, cls].sum() - tp)
        fn = int(matrix[cls, :].sum() - tp)
        union = tp + fp + fn
        rows.append(None if union == 0 else tp / union)
    tp, fp, fn = int(matrix[2, 2]), int(matrix[:, 2].sum() - matrix[2, 2]), int(matrix[2, :].sum() - matrix[2, 2])
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    defined_ious = [value for value in rows if value is not None]
    return {"confusion": matrix.tolist(), "pixels": int(matrix.sum()),
            "nonwater_iou": rows[0], "permanent_iou": rows[1], "flood_iou": rows[2],
            "macro_iou": None if not defined_ious else float(np.mean(defined_ious)),
            "flood_precision": precision, "flood_recall": recall, "flood_f1": f1}


def _process_ci(records: dict[str, dict], field: str, seed: int = 20261029) -> dict:
    values = [(process, item[field]) for process, item in sorted(records.items())
              if item[field] is not None]
    if not values:
        return {"n_processes": 0, "process_ids": [], "mean": None, "ci95": None}
    vector = np.asarray([value for _, value in values], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(10_000, dtype=np.float64)
    for index in range(len(draws)):
        draws[index] = vector[rng.integers(0, len(vector), size=len(vector))].mean()
    return {"n_processes": len(values), "process_ids": [key for key, _ in values],
            "mean": float(vector.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]}


def evaluate_b1(root: Path, config: dict, role: str) -> dict:
    if role not in ("internal_retro", "holdout_retro"):
        raise ValueError("B1 role must be internal_retro or holdout_retro")
    output = root / "outputs" / "upgrade_20260929" / "B1" / f"evaluation_{role}.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite a one-time B1 evaluation artifact: {output}")
    locked = require_real_access(root, config)
    baseline = config.get("fixed_baselines", {}).get("B1")
    if not baseline:
        raise RuntimeError("frozen config lacks the fixed B1 hard baseline")
    threshold_path = root / baseline["thresholds_path"]
    if sha256(threshold_path) != baseline["thresholds_sha256"]:
        raise RuntimeError("B1 threshold file hash mismatch")
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    if (threshold.get("model") != "B1" or threshold.get("status") != "FIT_ROLE_ONLY_COMPLETE"
            or threshold.get("fit_manifest_sha256") != config["manifests"]["fit"]["sha256"]
            or threshold.get("config_sha256") != sha256(root / "configs" / "upgrade_20260929.json")
            or threshold.get("fit_script_sha256") != baseline.get("fit_script_sha256")
            or threshold.get("nonfit_roles_read") != [] or threshold.get("optical_arrays_read") is not False):
        raise RuntimeError("B1 threshold provenance is not training-only")

    manifest = root / config["manifests"][role]["path"]
    grid_path = root / config["role_grids"][role]["path"]
    owners = manifest_owner_masks(manifest, grid_path, expected_role=role)
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = sorted(rows, key=lambda item: item["sample_id"])
    strata: dict[str, list[dict]]
    if role == "holdout_retro":
        aux = config["auxiliary_metadata"]
        for name in ("holdout_eligibility", "holdout_s1_anchor_audit_csv", "holdout_s1_anchor_audit_json"):
            entry = aux[name]
            if sha256(root / entry["path"]) != entry["sha256"]:
                raise RuntimeError(f"B1 frozen auxiliary metadata hash mismatch: {name}")
        with (root / aux["holdout_eligibility"]["path"]).open(newline="", encoding="utf-8") as f:
            eligibility = list(csv.DictReader(f))
        with (root / aux["holdout_s1_anchor_audit_csv"]["path"]).open(newline="", encoding="utf-8") as f:
            anchors = list(csv.DictReader(f))
        strata = classify_holdout_strata(rows, eligibility, anchors)
    else:
        strata = {"internal_retro": rows}

    archive = root / "data" / "raw" / "impactmesh" / ("train" if role == "internal_retro" else "test")
    results = {}
    with TrainArchiveReader(archive) as reader:
        for stratum, group_rows in strata.items():
            grouped: dict[str, list[dict]] = defaultdict(list)
            for row in group_rows:
                process, _ = owners[row["sample_id"]]
                grouped[process].append(row)
            process_results = {}
            for process, process_rows in sorted(grouped.items()):
                confusion = np.zeros((3, 3), dtype=np.int64)
                fallback_tiles = 0
                for row in sorted(process_rows, key=lambda item: item["sample_id"]):
                    # B1 is SAR-only: load exactly S1 and MASK for every stratum.
                    sar, label = reader.load_s1_label(row["sample_id"])
                    event_vh = sar[1]
                    before, source_index = pre_event_vh(reader, row["sample_id"])
                    fallback_tiles += int(source_index == -1)
                    pred = predict(event_vh, before - event_vh,
                                   float(threshold["water_threshold_db"]),
                                   float(threshold["change_threshold_db"]))
                    owner = owners[row["sample_id"]][1]
                    selected = owner & (label != 255)
                    if selected.any():
                        indices = label[selected].astype(np.int64) * 3 + pred[selected].astype(np.int64)
                        confusion += np.bincount(indices, minlength=9).reshape(3, 3)
                metrics = _hard_metrics(confusion)
                metrics["tiles"] = len(process_rows)
                metrics["pre_event_fallback_tiles"] = fallback_tiles
                process_results[process] = metrics
            metric_names = ("nonwater_iou", "permanent_iou", "flood_iou", "macro_iou",
                            "flood_precision", "flood_recall", "flood_f1")
            summaries = {name: _process_ci(process_results, name) for name in metric_names}
            results[stratum] = {"processes": process_results, "summary": summaries}
    payload = {"model": "B1", "role": role, "interpretation": "retrospective descriptive only",
               "metric_scope": "hard-label segmentation only; no probability or calibration metrics",
               "optical_states": "not applicable; fixed SAR-only rule evaluated once",
               "strata_not_pooled": role == "holdout_retro", "strata_counts": {k: len(v) for k, v in strata.items()},
               "thresholds": {"water_threshold_db": threshold["water_threshold_db"],
                              "change_threshold_db": threshold["change_threshold_db"]},
               "thresholds_sha256": sha256(threshold_path), "fit_manifest_sha256": threshold["fit_manifest_sha256"],
               "role_manifest_sha256": sha256(manifest),
               "grid_sha256": sha256(grid_path), "locked_input_hashes": locked,
               "config_sha256": sha256(root / "configs" / "upgrade_20260929.json"),
               "completed_utc": datetime.now(timezone.utc).isoformat(), "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    payload["output_path"] = str(output)
    return payload


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("internal_retro", "holdout_retro"))
    args = parser.parse_args()
    config = json.loads((ROOT / "configs" / "upgrade_20260929.json").read_text(encoding="utf-8"))
    result = evaluate_b1(ROOT, config, args.role)
    print(json.dumps({"model": "B1", "role": args.role, "strata": result["strata_counts"],
                      "output_path": result["output_path"]}, indent=2))


if __name__ == "__main__":
    main()
