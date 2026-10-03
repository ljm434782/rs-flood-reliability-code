"""Evaluate frozen hard-label B1 SAR threshold baseline on internal processes."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from flood_reliability.evaluation_counts import confusion_counts
from flood_reliability.formal_metrics import class_iou, flood_f1, macro_with_bootstrap
from flood_reliability.impactmesh_io import TrainArchiveReader
from flood_reliability.overlap import unique_pixel_mask
from scripts.calibrate_impactmesh_formal import CONFIG, ELIGIBILITY, EXCLUSIONS, GRID, ROOT, SPLIT, role_rows, sha256
from scripts.fit_impactmesh_b1 import pre_event_vh, predict


def main():
    for model in ("B2", "B4", "B5", "P"):
        folder = ROOT / "outputs" / "formal_v1" / model
        if not (folder / "run.json").exists() or not (folder / "temperature.json").exists():
            raise RuntimeError("Do not open internal test before all neural models/calibration freeze")
    source = ROOT / "outputs" / "formal_v1" / "B1" / "thresholds.json"
    fit = json.loads(source.read_text(encoding="utf-8"))
    if (fit["split_sha256"] != sha256(SPLIT)
            or fit["eligibility_sha256"] != sha256(ELIGIBILITY)
            or fit["config_sha256"] != sha256(CONFIG)):
        raise RuntimeError("B1 thresholds not fitted to frozen training roles")
    groups, grids = role_rows("internal_test")
    process_rows = {}
    with TrainArchiveReader(ROOT / "data" / "raw" / "impactmesh" / "train") as reader:
        for index, (process, rows) in enumerate(groups.items(), 1):
            confusion = np.zeros((3, 3), dtype=np.int64)
            preceding = []
            pre_missing = 0
            for row in rows:
                sar, _, _, label = reader.load(row["sample_id"], int(row["selected_time_index"]))
                before, source_index = pre_event_vh(reader, row["sample_id"])
                pre_missing += source_index == -1
                prediction = predict(sar[1], before - sar[1],
                                     fit["water_threshold_db"], fit["change_threshold_db"])
                owner = unique_pixel_mask(grids[row["sample_id"]], preceding)
                preceding.append(grids[row["sample_id"]])
                confusion += confusion_counts(label, prediction, num_classes=3, evaluation_mask=owner)
            ious = [class_iou(confusion, k) for k in range(3)]
            process_rows[process] = {
                "tiles": len(rows), "pre_event_fallback_tiles": pre_missing,
                "pixels": int(confusion.sum()), "confusion": confusion.tolist(),
                "nonwater_iou": ious[0], "permanent_iou": ious[1],
                "flood_iou": ious[2],
                "macro_iou": float(np.mean([value for value in ious if value is not None])),
                "flood_f1": flood_f1(confusion),
            }
            print(f"B1 internal test process {index}/{len(groups)} {process}: "
                  f"{len(rows)} tiles", flush=True)
    metrics = ("nonwater_iou", "permanent_iou", "flood_iou", "macro_iou", "flood_f1")
    result = {
        "model": "B1", "role": "internal_test", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": sha256(source), "split_sha256": sha256(SPLIT),
        "eligibility_sha256": sha256(ELIGIBILITY), "grid_sha256": sha256(GRID),
        "evaluation_exclusions_sha256": sha256(EXCLUSIONS),
        "script_sha256": sha256(Path(__file__)), "processes": process_rows,
        "summary": {key: macro_with_bootstrap(process_rows, key) for key in metrics},
    }
    output = ROOT / "outputs" / "formal_v1" / "B1" / "internal_test.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
