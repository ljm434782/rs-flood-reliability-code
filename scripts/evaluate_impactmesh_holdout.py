"""Final official ImpactMesh process-out holdout evaluation after model freeze."""

import argparse
import csv
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
from scripts.calibrate_impactmesh_formal import CONFIG, EXCLUSIONS, ROOT, STATS, load_model, sha256
from scripts.evaluate_impactmesh_formal import METRICS, ScoreAccumulator
from scripts.fit_impactmesh_b1 import pre_event_vh, predict
from scripts.train_impactmesh_formal import encode_batch


MANIFESTS = ROOT / "data" / "manifests"
ELIGIBILITY = MANIFESTS / "impactmesh_holdout_frozen_eligibility.csv"
GRID = MANIFESTS / "impactmesh_test_mask_grid_index.csv"
RAW = ROOT / "data" / "raw" / "impactmesh" / "test"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("B1", "B2", "B4", "B5", "P"))
    parser.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    args = parser.parse_args()
    for model_name in ("B2", "B4", "B5", "P"):
        folder = ROOT / "outputs" / "formal_v1" / model_name
        if not (folder / "run.json").exists() or not (folder / "temperature.json").exists():
            raise RuntimeError("Holdout test requires all models and temperatures frozen")
    meta = MANIFESTS / "impactmesh_holdout_frozen_eligibility_meta.json"
    if not meta.exists():
        raise RuntimeError("Frozen input eligibility must be applied before holdout labels")
    audit = json.loads(meta.read_text(encoding="utf-8"))
    if audit["output_sha256"] != sha256(ELIGIBILITY) or audit["test_label_pixels_read"]:
        raise RuntimeError("Holdout input audit provenance mismatch")
    with EXCLUSIONS.open(newline="", encoding="utf-8") as handle:
        excluded = {row["activation"] for row in csv.DictReader(handle)
                    if row["original_role"] == "official_test_holdout"}
    if excluded != {"EMSR757", "EMSR763", "EMSR766"}:
        raise RuntimeError("Holdout same-storm exclusions changed")
    with ELIGIBILITY.open(newline="", encoding="utf-8") as handle:
        input_rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in input_rows:
        if row["paired_eligible"] == "1" and row["activation"] not in excluded:
            groups[row["activation"]].append(row)
    groups = {activation: sorted(rows, key=lambda row: row["sample_id"])
              for activation, rows in sorted(groups.items()) if len(rows) >= 5}
    if not groups:
        raise RuntimeError("No independent official holdout process reaches frozen minimum five paired tiles")
    with GRID.open(newline="", encoding="utf-8") as handle:
        grids = {row["sample_id"]: row for row in csv.DictReader(handle) if row["holdout"] == "1"}
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = None
    checkpoint = None
    temperature = 1.
    temperature_path = None
    threshold_path = None
    if args.model == "B1":
        threshold_path = ROOT / "outputs" / "formal_v1" / "B1" / "thresholds.json"
        fit = json.loads(threshold_path.read_text(encoding="utf-8"))
    else:
        model, checkpoint = load_model(args.model, device, stats, config)
        temperature_path = ROOT / "outputs" / "formal_v1" / args.model / "temperature.json"
        record = json.loads(temperature_path.read_text(encoding="utf-8"))
        if record["checkpoint_sha256"] != sha256(checkpoint):
            raise RuntimeError("Temperature/checkpoint mismatch")
        temperature = float(record["temperature"])
    results = {}
    with TrainArchiveReader(RAW) as reader:
        for process_index, (process, rows) in enumerate(groups.items(), 1):
            preceding = []
            if args.model == "B1":
                confusion = np.zeros((3, 3), dtype=np.int64)
                for row in rows:
                    sar, _, _, target = reader.load(row["sample_id"], int(row["selected_time_index"]))
                    before, _ = pre_event_vh(reader, row["sample_id"])
                    prediction = predict(sar[1], before - sar[1],
                                         fit["water_threshold_db"], fit["change_threshold_db"])
                    owner = unique_pixel_mask(grids[row["sample_id"]], preceding)
                    preceding.append(grids[row["sample_id"]])
                    confusion += confusion_counts(target, prediction, num_classes=3,
                                                   evaluation_mask=owner)
                ious = [class_iou(confusion, k) for k in range(3)]
                result = {"pixels": int(confusion.sum()), "confusion": confusion.tolist(),
                          "nonwater_iou": ious[0], "permanent_iou": ious[1],
                          "flood_iou": ious[2], "flood_f1": flood_f1(confusion),
                          "macro_iou": float(np.mean([v for v in ious if v is not None]))}
            else:
                raw = ScoreAccumulator()
                calibrated = ScoreAccumulator()
                for start in range(0, len(rows), config["batch_size"]):
                    batch = rows[start:start + config["batch_size"]]
                    sar, optical, valid, target, offset = encode_batch(reader, batch, stats, device)
                    present = torch.ones(len(batch), dtype=torch.bool, device=device)
                    with torch.inference_mode():
                        logits_t = (model(sar) if args.model == "B2" else
                                    model(sar, optical, valid, present,
                                          normalized_time_offset=offset))
                    logits = logits_t.cpu().numpy()
                    labels = target.cpu().numpy()
                    for i, row in enumerate(batch):
                        owner = unique_pixel_mask(grids[row["sample_id"]], preceding)
                        preceding.append(grids[row["sample_id"]])
                        raw.add(labels[i], logits[i].copy(), owner, 1., collect_ap=True)
                        calibrated.add(labels[i], logits[i].copy(), owner, temperature, collect_ap=True)
                result = {"raw": raw.finish(), "calibrated": calibrated.finish()}
            results[process] = {"tiles": len(rows), "scores": result}
            print(f"{args.model} official holdout process {process_index}/{len(groups)} "
                  f"{process}: {len(rows)} tiles", flush=True)
    if args.model == "B1":
        keys = ("flood_iou", "flood_f1", "permanent_iou", "macro_iou")
        summary = {key: macro_with_bootstrap(
            {p: record["scores"] for p, record in results.items()}, key) for key in keys}
    else:
        summary = {state: {key: macro_with_bootstrap(
            {p: record["scores"][state] for p, record in results.items()}, key)
            for key in METRICS} for state in ("raw", "calibrated")}
    output = ROOT / "outputs" / "formal_v1" / args.model / "official_holdout.json"
    output.write_text(json.dumps({
        "model": args.model, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "independent_eligible_process_count": len(groups),
        "excluded_same_storm_activations": sorted(excluded),
        "eligibility_sha256": sha256(ELIGIBILITY), "eligibility_meta_sha256": sha256(meta),
        "grid_sha256": sha256(GRID), "event_exclusions_sha256": sha256(EXCLUSIONS),
        "checkpoint_sha256": None if checkpoint is None else sha256(checkpoint),
        "temperature_sha256": None if temperature_path is None else sha256(temperature_path),
        "thresholds_sha256": None if threshold_path is None else sha256(threshold_path),
        "script_sha256": sha256(Path(__file__)), "processes": results, "summary": summary,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
