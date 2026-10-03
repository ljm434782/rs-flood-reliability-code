"""Evaluate S1 tiles with no S2 satisfying the frozen pre-event optical rule."""

import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from flood_reliability.evaluation_counts import confusion_counts
from flood_reliability.formal_metrics import class_iou, flood_f1, macro_with_bootstrap
from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.overlap import unique_pixel_mask
from scripts.calibrate_impactmesh_formal import (
    CONFIG, ELIGIBILITY, EXCLUSIONS, GRID, ROOT, SPLIT, STATS, load_model, sha256,
)
from scripts.evaluate_impactmesh_formal import METRICS, ScoreAccumulator
from scripts.fit_impactmesh_b1 import pre_event_vh, predict


REASONS = ("visible_below_0.5", "no_pre_s1_s2_within_30d")


def missing_rows():
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        roles = {row["activation"]: row for row in csv.DictReader(handle)}
    with EXCLUSIONS.open(newline="", encoding="utf-8") as handle:
        excluded = {row["activation"] for row in csv.DictReader(handle)}
    groups = defaultdict(list)
    with gzip.open(ELIGIBILITY, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            activation = roles[row["activation"]]
            if (activation["role"] == "internal_test" and row["activation"] not in excluded
                    and row["paired_eligible"] == "0" and row["failure_reasons"] in REASONS):
                groups[activation["process_id"]].append(row)
    for values in groups.values():
        values.sort(key=lambda row: row["sample_id"])
    return dict(sorted(groups.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("B1", "B2", "B4", "B5", "P"))
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    args = parser.parse_args()
    for model_name in ("B2", "B4", "B5", "P"):
        folder = ROOT / "outputs" / "formal_v1" / model_name
        if not (folder / "run.json").exists() or not (folder / "temperature.json").exists():
            raise RuntimeError("Real-missing final test requires frozen models/calibration")
    groups = missing_rows()
    if not groups:
        raise RuntimeError("No real missing S2 test samples after frozen filtering")
    with GRID.open(newline="", encoding="utf-8") as handle:
        grids = {row["sample_id"]: row for row in csv.DictReader(handle) if row["split"] == "train"}
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = None
    checkpoint = None
    temperature_path = None
    threshold_path = None
    temperature = 1.
    if args.model == "B1":
        threshold_path = ROOT / "outputs" / "formal_v1" / "B1" / "thresholds.json"
        baseline = json.loads(threshold_path.read_text(encoding="utf-8"))
    else:
        model, checkpoint = load_model(args.model, device, stats, config)
        temperature_path = ROOT / "outputs" / "formal_v1" / args.model / "temperature.json"
        temp_record = json.loads(temperature_path.read_text(encoding="utf-8"))
        if temp_record["checkpoint_sha256"] != sha256(checkpoint):
            raise RuntimeError("Temperature/checkpoint mismatch")
        temperature = float(temp_record["temperature"])
    means = np.asarray(stats["sar_mean"], dtype=np.float32)[:, None, None]
    stds = np.asarray(stats["sar_std"], dtype=np.float32)[:, None, None]
    result = {}
    with TrainArchiveReader(ROOT / "data" / "raw" / "impactmesh" / "train") as reader:
        for category in ("all_unqualified", *REASONS):
            process_scores = {}
            no_valid_processes = []
            for process, all_rows in groups.items():
                rows = (all_rows if category == "all_unqualified" else
                        [row for row in all_rows if row["failure_reasons"] == category])
                if not rows:
                    continue
                preceding = []
                if args.model == "B1":
                    confusion = np.zeros((3, 3), np.int64)
                else:
                    raw, calibrated = ScoreAccumulator(), ScoreAccumulator()
                for row in rows:
                    sar, target = reader.load_s1_label(row["sample_id"])
                    owner = unique_pixel_mask(grids[row["sample_id"]], preceding)
                    preceding.append(grids[row["sample_id"]])
                    if args.model == "B1":
                        before, _ = pre_event_vh(reader, row["sample_id"])
                        guess = predict(sar[1], before - sar[1],
                                        baseline["water_threshold_db"], baseline["change_threshold_db"])
                        confusion += confusion_counts(target, guess, num_classes=3,
                                                       evaluation_mask=owner)
                    else:
                        normalized = torch.from_numpy(((sar - means) / stds)[None]).to(device)
                        with torch.inference_mode():
                            if args.model == "B2":
                                logits = model(normalized)
                            else:
                                optical = torch.zeros((1, 12, 256, 256), device=device)
                                valid = torch.zeros((1, 1, 256, 256), dtype=torch.bool, device=device)
                                present = torch.zeros(1, dtype=torch.bool, device=device)
                                offset = torch.zeros(1, device=device)
                                logits = model(normalized, optical, valid, present,
                                               normalized_time_offset=offset)
                        values = logits[0].cpu().numpy()
                        raw.add(target, values.copy(), owner, 1., collect_ap=True)
                        calibrated.add(target, values.copy(), owner, temperature, collect_ap=True)
                if args.model == "B1":
                    if confusion.sum() == 0:
                        no_valid_processes.append(process)
                        continue
                    ious = [class_iou(confusion, k) for k in range(3)]
                    scores = {"pixels": int(confusion.sum()), "confusion": confusion.tolist(),
                              "nonwater_iou": ious[0], "permanent_iou": ious[1],
                              "flood_iou": ious[2], "flood_f1": flood_f1(confusion),
                              "macro_iou": float(np.mean([v for v in ious if v is not None]))}
                else:
                    if raw.pixels == 0:
                        no_valid_processes.append(process)
                        continue
                    scores = {"raw": raw.finish(), "calibrated": calibrated.finish()}
                process_scores[process] = {"tiles": len(rows), "scores": scores}
            if args.model == "B1":
                keys = ("flood_iou", "flood_f1", "permanent_iou", "macro_iou")
                summary = {key: macro_with_bootstrap(
                    {p: row["scores"] for p, row in process_scores.items()}, key) for key in keys}
            else:
                summary = {state: {key: macro_with_bootstrap(
                    {p: row["scores"][state] for p, row in process_scores.items()}, key)
                    for key in METRICS} for state in ("raw", "calibrated")}
            result[category] = {"process_count": len(process_scores),
                                "tile_count": sum(row["tiles"] for row in process_scores.values()),
                                "no_valid_label_processes": no_valid_processes,
                                "processes": process_scores, "summary": summary}
            print(f"{args.model} real missing {category}: {len(process_scores)} processes, "
                  f"{result[category]['tile_count']} tiles", flush=True)
    output = ROOT / "outputs" / "formal_v1" / args.model / "real_missing_s2.json"
    output.write_text(json.dumps({
        "model": args.model, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "independent internal-test S1 tiles with no S2 meeting frozen pre-S1 rules",
        "eligibility_sha256": sha256(ELIGIBILITY), "split_sha256": sha256(SPLIT),
        "event_exclusions_sha256": sha256(EXCLUSIONS), "grid_sha256": sha256(GRID),
        "checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
        "temperature_sha256": None if temperature_path is None else sha256(temperature_path),
        "thresholds_sha256": None if threshold_path is None else sha256(threshold_path),
        "script_sha256": sha256(Path(__file__)), "categories": result,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
