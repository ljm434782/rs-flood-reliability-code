"""Diagnostic distance from ImpactMesh unseen activations to Sen1 hand test.

Spatial distance is only a screen for external-test leakage. It cannot prove
distinct flood processes or source-image independence.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pyproj import Transformer
from shapely.ops import transform

from audit_impactmesh_sen1_overlap import grid_footprint, label_footprint, load_grids, verified_sen_labels


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
OUTPUT = MANIFESTS / "impactmesh_holdout_external_distance.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    holdout_file = MANIFESTS / "impactmesh_flood_test_holdout.txt"
    sample_ids = holdout_file.read_text(encoding="utf-8").splitlines()
    if len(sample_ids) != 562 or len(set(sample_ids)) != 562:
        raise RuntimeError("Unexpected official holdout sample count or duplicates")
    labels = verified_sen_labels()
    grids, grid_hashes = load_grids()
    to_equal_area = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform
    by_activation = defaultdict(list)
    for sample_id in sample_ids:
        grid = grids[sample_id]
        if grid["split"] != "test" or grid["holdout"] != "1":
            raise RuntimeError(f"Holdout ID disagrees with actual grid index: {sample_id}")
        activation, polygon = grid["activation"], grid_footprint(grid)
        by_activation[activation].append((sample_id, transform(to_equal_area, polygon)))
    by_event = defaultdict(list)
    for path in labels:
        scene = path.name.removesuffix("_LabelHand.tif")
        event = scene.split("_", 1)[0]
        by_event[event].append((scene, transform(to_equal_area, label_footprint(path))))
    rows = []
    for activation, chips in sorted(by_activation.items()):
        nearest = min(
            (chip.distance(scene_poly), event, sample_id, scene)
            for event, scenes in by_event.items()
            for sample_id, chip in chips
            for scene, scene_poly in scenes
        )
        rows.append({
            "activation": activation,
            "holdout_sample_count": len(chips),
            "nearest_sen1_event": nearest[1],
            "nearest_holdout_sample": nearest[2],
            "nearest_sen1_scene": nearest[3],
            "projected_minimum_gap_km": round(nearest[0] / 1000, 3),
        })
    output = {
        "holdout_id_list_sha256": digest(holdout_file),
        "mask_grid_index_sha256": grid_hashes,
        "sen1_manifest_sha256": digest(MANIFESTS / "sen1floods11_manifest.csv"),
        "holdout_activation_count": len(rows),
        "holdout_sample_count": len(sample_ids),
        "sen1_event_count": len(by_event),
        "sen1_scene_count": len(labels),
        "distance_crs": "EPSG:6933",
        "distance_limit": "Equal-area projected polygon gap is diagnostic, not precise geodesic distance or proof of process/source independence. ImpactMesh bounds and CRS come from verified official TIFF headers; Sen1 bounds are read from test labels. Test label pixel values are not read.",
        "rows": rows,
    }
    if len(rows) != 12 or len(by_event) != 10:
        raise RuntimeError("Unexpected holdout activation or Sen1 event count")
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}; closest diagnostic gap {min(row['projected_minimum_gap_km'] for row in rows):.3f} km")


if __name__ == "__main__":
    main()
