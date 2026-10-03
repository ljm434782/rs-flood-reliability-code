"""Fit a simple two-threshold SAR change baseline using training roles only."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flood_reliability.formal_metrics import class_iou
from flood_reliability.impactmesh_io import TrainArchiveReader, _group
from scripts.train_impactmesh_formal import CONFIG, ELIGIBILITY, ROOT, SPLIT, prepare_rows, sha256


def pre_event_vh(reader: TrainArchiveReader, sample_id: str):
    group = _group(reader.blob("S1RTC", sample_id))
    times = group.attrs["datetime"].split(";")
    if len(times) != 4:
        raise RuntimeError(f"Unexpected S1 acquisitions: {sample_id}")
    event = np.asarray(group["bands"][2, 1], dtype=np.float32)
    for index in (1, 0):
        if (times[index] < times[2] and int(group["data_mask"][index]) == 1
                and not np.any(group["nan_mask"][index] != 0)):
            before = np.asarray(group["bands"][index, 1], dtype=np.float32)
            if np.isfinite(before).all():
                return before, index
    return event, -1


def predict(event_vh: np.ndarray, change: np.ndarray,
            water_threshold: float, change_threshold: float) -> np.ndarray:
    water = event_vh <= water_threshold
    flood = water & (change >= change_threshold)
    result = np.zeros(event_vh.shape, dtype=np.uint8)
    result[water] = 1
    result[flood] = 2
    return result


def process_macro_iou(truth: np.ndarray, prediction: np.ndarray) -> float:
    confusion = np.bincount(truth.astype(np.int64) * 3 + prediction.astype(np.int64),
                            minlength=9).reshape(3, 3)
    valid = [class_iou(confusion, index) for index in range(3)]
    return float(np.mean([value for value in valid if value is not None]))


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    groups = prepare_rows(config)["train"]
    if len(groups) != 81:
        raise RuntimeError("Unexpected training process count")
    by_process = {}
    pre_counts = {"index_1": 0, "index_0": 0, "missing_fallback_event": 0}
    with TrainArchiveReader(ROOT / "data" / "raw" / "impactmesh" / "train") as reader:
        for position, process in enumerate(sorted(groups), 1):
            observations = []
            for row in groups[process][:config["normalization_chips_per_process"]]:
                sar, _, _, label = reader.load(row["sample_id"], int(row["selected_time_index"]))
                before, source_index = pre_event_vh(reader, row["sample_id"])
                pre_counts[{1: "index_1", 0: "index_0", -1: "missing_fallback_event"}[source_index]] += 1
                event_vh = sar[1, ::4, ::4]
                change = before[::4, ::4] - event_vh
                truth = label[::4, ::4]
                selected = truth != 255
                observations.append((event_vh[selected], change[selected], truth[selected]))
            by_process[process] = tuple(np.concatenate([part[i] for part in observations])
                                        for i in range(3))
            if position % 20 == 0:
                print(f"B1 train-only threshold data: {position}/{len(groups)} processes", flush=True)
    all_event = np.concatenate([values[0] for values in by_process.values()])
    all_change = np.concatenate([values[1] for values in by_process.values()])
    water_grid = np.unique(np.percentile(all_event, np.arange(5, 65, 5)))
    change_grid = np.unique(np.maximum(0, np.percentile(all_change, np.arange(10, 100, 10))))
    if not len(water_grid) or not len(change_grid):
        raise RuntimeError("Empty train-derived SAR threshold grid")
    best = None
    for water in water_grid:
        for delta in change_grid:
            scores = []
            for event, change, truth in by_process.values():
                scores.append(process_macro_iou(truth, predict(event, change, water, delta)))
            candidate = (float(np.mean(scores)), -float(water), float(delta))
            if best is None or candidate > best[0]:
                best = (candidate, float(water), float(delta))
    output = ROOT / "outputs" / "formal_v1" / "B1"
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "model": "B1", "rule": "event VH <= water threshold; among water, prior VH minus event VH >= change threshold -> flood, else permanent water",
        "pre_event_choice": "most recent valid earlier S1 index 1, else 0, else event VH as no-change fallback",
        "training_pixel_sample": "fixed first four hash-order tiles per training process, every fourth row and column; Ignore excluded",
        "train_process_macro_iou": best[0][0],
        "water_threshold_db": best[1], "change_threshold_db": best[2],
        "water_grid_db": water_grid.tolist(), "change_grid_db": change_grid.tolist(),
        "pre_event_source_counts": pre_counts,
        "training_process_count": len(groups),
        "selected_training_pixels": int(sum(len(x[2]) for x in by_process.values())),
        "fitted_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": sha256(SPLIT), "eligibility_sha256": sha256(ELIGIBILITY),
        "config_sha256": sha256(CONFIG), "script_sha256": sha256(Path(__file__)),
        "fit_limit": "training roles only; no calibration, validation, internal or external test labels",
    }
    (output / "thresholds.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"water_threshold_db": best[1], "change_threshold_db": best[2],
                      "train_process_macro_iou": best[0][0]}), flush=True)


if __name__ == "__main__":
    main()
