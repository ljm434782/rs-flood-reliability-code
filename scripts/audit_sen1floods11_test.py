"""Verify raster pairing and label values in the official Sen1Floods11 hand test split."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "raw" / "sen1floods11" / "data" / "flood_events" / "HandLabeled"
OUTPUT = ROOT / "data" / "manifests" / "sen1floods11_test_audit.csv"


def main() -> None:
    rows = []
    for label_path in sorted((BASE / "LabelHand").glob("*.tif")):
        s1_path = BASE / "S1Hand" / label_path.name.replace("_LabelHand.tif", "_S1Hand.tif")
        if not s1_path.exists():
            raise RuntimeError(f"Missing S1 mate for {label_path.name}")
        with rasterio.open(label_path) as label, rasterio.open(s1_path) as s1:
            values, counts = np.unique(label.read(1), return_counts=True)
            classes = {str(value): int(count) for value, count in zip(values.tolist(), counts.tolist())}
            if not set(classes).issubset({"-1", "0", "1"}):
                raise RuntimeError(f"Unexpected labels in {label_path.name}: {classes}")
            transform_tolerance = min(abs(label.transform.a), abs(label.transform.e)) * 0.001
            grids_match = (
                s1.crs == label.crs
                and s1.shape == label.shape
                and np.allclose(tuple(s1.transform), tuple(label.transform), atol=transform_tolerance, rtol=0)
            )
            if not grids_match:
                raise RuntimeError(f"Raster grid mismatch: {label_path.name}")
            rows.append({
                "scene": label_path.name.removesuffix("_LabelHand.tif"),
                "event": label_path.name.split("_", 1)[0],
                "height": label.height,
                "width": label.width,
                "crs": str(label.crs),
                "s1_bands": s1.count,
                "label_nodata": label.nodata,
                "count_-1": classes.get("-1", 0),
                "count_0": classes.get("0", 0),
                "count_1": classes.get("1", 0),
                "s1_grid_matches_label": True,
            })
    if len(rows) != 90:
        raise RuntimeError(f"Expected 90 official hand test pairs, found {len(rows)}")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} S1/Label hand test raster pairs from {len(Counter(row['event'] for row in rows))} events")
    print("Total label pixels:", {key: sum(row[f"count_{key}"] for row in rows) for key in ("-1", "0", "1")})
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
