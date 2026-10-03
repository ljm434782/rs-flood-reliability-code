"""Fit B1 SAR thresholds using only the frozen G3 fit role.

The search is deterministic and bounded. It reads S1 and MASK only from the
fit manifest; it does not access optical arrays, calibration labels, or any
evaluation role.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from flood_reliability.impactmesh_io import TrainArchiveReader
from scripts.fit_impactmesh_b1 import predict, pre_event_vh, process_macro_iou
from scripts.upgrade_calibrate_evaluate import manifest_owner_masks, require_real_access, sha256


def fit_b1(root: Path, config: dict) -> dict:
    output = root / config["fixed_baselines"]["B1"]["thresholds_path"]
    if output.exists():
        raise RuntimeError(f"refusing to overwrite a B1 fit artifact: {output}")
    locked = require_real_access(root, config)
    manifest_item = config["manifests"]["fit"]
    manifest = root / manifest_item["path"]
    grid = root / config["role_grids"]["fit"]["path"]
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    if any(row.get("derived_role") != "fit" for row in rows) or len(rows) != 3525:
        raise RuntimeError("B1 fit requires exactly the frozen 3,525-row fit role")
    owners = manifest_owner_masks(manifest, grid, expected_role="fit")
    selected: dict[str, list[dict]] = {}
    for row in rows:
        selected.setdefault(row["conservative_unit_id"], []).append(row)
    for unit in selected:
        selected[unit] = sorted(selected[unit], key=lambda row: row["sample_id"])[:4]

    per_unit = {}
    with TrainArchiveReader(root / "data" / "raw" / "impactmesh" / "train") as reader:
        for unit in sorted(selected):
            event, delta, truth = [], [], []
            preceding = []
            for row in selected[unit]:
                sid = row["sample_id"]
                sar, label = reader.load_s1_label(sid)
                before, _ = pre_event_vh(reader, sid)
                owner = owners[sid][1]
                # manifest_owner_masks already assigns a single deterministic
                # owner to overlapping tiles within each unit.
                event_grid = sar[1, ::4, ::4]
                delta_grid = (before - sar[1])[::4, ::4]
                truth_grid = label[::4, ::4]
                chosen = owner[::4, ::4] & (truth_grid != 255)
                event.append(event_grid[chosen]); delta.append(delta_grid[chosen]); truth.append(truth_grid[chosen])
            if not event or sum(len(part) for part in truth) == 0:
                raise RuntimeError(f"B1 fit unit has no owned sampled pixels: {unit}")
            per_unit[unit] = (np.concatenate(event), np.concatenate(delta), np.concatenate(truth))

    all_event = np.concatenate([values[0] for values in per_unit.values()])
    all_delta = np.concatenate([values[1] for values in per_unit.values()])
    water_grid = np.unique(np.percentile(all_event, np.arange(5, 65, 5)))
    change_grid = np.unique(np.maximum(0, np.percentile(all_delta, np.arange(10, 100, 10))))
    best = None
    for water_threshold in water_grid:
        for change_threshold in change_grid:
            per_unit_scores = [process_macro_iou(
                truth, predict(event, delta, float(water_threshold), float(change_threshold)))
                for event, delta, truth in per_unit.values()]
            candidate = (float(np.mean(per_unit_scores)), -float(water_threshold), float(change_threshold))
            if best is None or candidate > best[0]:
                best = (candidate, float(water_threshold), float(change_threshold))
    if best is None:
        raise RuntimeError("B1 fit threshold search had no candidates")
    payload = {"model": "B1", "status": "FIT_ROLE_ONLY_COMPLETE",
               "rule": "event VH <= water threshold; among water, prior VH minus event VH >= change threshold -> flood, else permanent water",
               "pre_event_choice": "most recent earlier valid S1 index 1, else 0, else event VH as no-change fallback",
               "training_role": "g3_rule_a_derived_candidate_fit only; 60 potential-dependence units",
               "tile_selection": "first four sample IDs in ascending lexical order per conservative unit",
               "pixel_selection": "every fourth row and column after deterministic unique-pixel ownership; Ignore excluded",
               "fit_objective": "mean three-class macro IoU, equally weighted over fit units",
               "tie_rule": "lower water threshold, then higher change threshold",
               "water_threshold_db": best[1], "change_threshold_db": best[2],
               "train_unit_macro_iou": best[0][0], "fit_unit_count": len(per_unit),
               "selected_tile_count": sum(len(value) for value in selected.values()),
               "selected_pixel_count": int(sum(len(value[0]) for value in per_unit.values())),
               "water_grid_db": water_grid.tolist(), "change_grid_db": change_grid.tolist(),
               "fit_manifest_sha256": manifest_item["sha256"], "fit_grid_sha256": config["role_grids"]["fit"]["sha256"],
               "config_sha256": sha256(root / "configs" / "upgrade_20260929.json"),
               "fit_script_sha256": sha256(Path(__file__)), "locked_inputs": locked,
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "nonfit_roles_read": [], "optical_arrays_read": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    payload["output_path"] = str(output)
    return payload


def main() -> None:
    config = json.loads((ROOT / "configs" / "upgrade_20260929.json").read_text(encoding="utf-8"))
    output = ROOT / "outputs" / "upgrade_20260929" / "B1" / "thresholds.json"
    expected = {"fit_manifest_sha256": config["manifests"]["fit"]["sha256"],
                "config_sha256": sha256(ROOT / "configs" / "upgrade_20260929.json"),
                "fit_script_sha256": sha256(Path(__file__))}
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in expected.items()):
            raise RuntimeError("existing B1 fit artifact differs; refusing overwrite")
        print(json.dumps({"status": existing["status"], "output_path": str(output), "resumed": True}, indent=2))
        return
    result = fit_b1(ROOT, config)
    print(json.dumps({"status": result["status"], "water_threshold_db": result["water_threshold_db"],
                      "change_threshold_db": result["change_threshold_db"],
                      "output_path": result["output_path"]}, indent=2))


if __name__ == "__main__":
    main()
