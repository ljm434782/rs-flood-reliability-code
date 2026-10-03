"""Compare blind candidate label-grid footprints with other official activations.

Geometry metadata only. This does not read label pixels or model predictions.
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform
from shapely.strtree import STRtree

from scripts.train_impactmesh_formal import ROOT, sha256

MAN = ROOT / "data/manifests"
OUT = MAN / "impactmesh_new_blind_spatial_screen.csv"
META = OUT.with_suffix(".json")
PREQUALIFIED = {"EMSR410", "EMSR411", "EMSR496", "EMSR559", "EMSR788", "EMSR843"}


def rows(split):
    path = MAN / f"impactmesh_{split}_mask_grid_index.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def polygon(row, transformers):
    crs = row["grid_crs"]
    if crs not in transformers:
        transformers[crs] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    return transform(transformers[crs], box(*(float(row[x]) for x in ("left", "bottom", "right", "top"))))


def main():
    if OUT.exists() or META.exists():
        raise RuntimeError("Refusing to overwrite candidate spatial screen")
    source = rows("val")
    candidate = [row for row in source if row["activation"] in PREQUALIFIED]
    comparison = [row for row in rows("train") + rows("test")
                  if row["activation"] not in PREQUALIFIED]
    transformers = {}
    geometry = [polygon(row, transformers) for row in comparison]
    tree = STRtree(geometry)
    outputs = []
    for row in candidate:
        shape = polygon(row, transformers)
        overlap = Counter()
        for index in tree.query(shape, predicate="intersects"):
            if shape.intersection(geometry[index]).area > 0:
                overlap[comparison[index]["activation"]] += 1
        outputs.append({"sample_id": row["sample_id"], "activation": row["activation"],
                        "overlapping_other_activation_tile_count": sum(overlap.values()),
                        "overlapping_other_activation_ids": ";".join(sorted(overlap))})
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(outputs[0]))
        writer.writeheader()
        writer.writerows(outputs)
    by_event = defaultdict(set)
    for row in outputs:
        by_event[row["activation"]].update(filter(None, row["overlapping_other_activation_ids"].split(";")))
    META.write_text(json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "candidate geometry metadata spatial overlap only",
        "candidate_tiles": len(outputs),
        "overlap_activation_ids_by_candidate": {k: sorted(v) for k, v in sorted(by_event.items())},
        "source_sha256": {s: sha256(MAN / f"impactmesh_{s}_mask_grid_index.csv")
                          for s in ("train", "val", "test")},
        "output_sha256": sha256(OUT), "label_pixels_read": False,
        "model_outputs_read": False,
        "limitation": "geometric overlap alone does not establish a shared flood event"},
        indent=2) + "\n", encoding="utf-8")
    print(META.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
