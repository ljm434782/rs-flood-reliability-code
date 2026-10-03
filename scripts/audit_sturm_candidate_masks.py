"""Audit grid overlap and mask agreement for all STURM same-name candidates."""

import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "manifests" / "sturm_candidate_pairs.csv"
MASKS = ROOT / "data" / "raw" / "sturm" / "candidate_masks"
OUTPUT = ROOT / "data" / "manifests" / "sturm_candidate_mask_audit.csv"
SUMMARY = ROOT / "data" / "manifests" / "sturm_candidate_mask_summary.csv"


def audit_pair(row: dict[str, str]) -> dict[str, object]:
    tile_id = row["tile_id"]
    with rasterio.open(MASKS / "S1" / tile_id) as s1, rasterio.open(MASKS / "S2" / tile_id) as s2:
        labels_s1 = s1.read(1)
        labels_s2 = np.full(labels_s1.shape, 255, dtype=np.int32)
        reproject(
            source=s2.read(1).astype(np.int32),
            destination=labels_s2,
            src_transform=s2.transform,
            src_crs=s2.crs,
            src_nodata=s2.nodata,
            dst_transform=s1.transform,
            dst_crs=s1.crs,
            dst_nodata=255,
            resampling=Resampling.nearest,
        )
        valid_s1 = labels_s1 != s1.nodata
        valid_common = valid_s1 & (labels_s2 != 255)
        common = int(valid_common.sum())
        flood_s1 = (labels_s1 == 1) & valid_common
        flood_s2 = (labels_s2 == 1) & valid_common
        union = int((flood_s1 | flood_s2).sum())
        result = {
            "tile_id": tile_id,
            "ems_code": row["ems_code_s1"],
            "aoi_code": row["aoi_code_s1"],
            "same_floodmap_id": row["same_floodmap_id"],
            "floodmap_id_s1": row["floodmap_id_s1"],
            "floodmap_id_s2": row["floodmap_id_s2"],
            "floodmap_date_s1": row["floodmap_date_s1"],
            "floodmap_date_s2": row["floodmap_date_s2"],
            "s1_time": row["s1_time"],
            "s2_time": row["s2_time"],
            "delta_s2_minus_s1_days": row["delta_s2_minus_s1_days"],
            "s1_crs": str(s1.crs),
            "s2_crs": str(s2.crs),
            "same_transform": s1.transform == s2.transform and s1.crs == s2.crs,
            "s1_valid_pixels": int(valid_s1.sum()),
            "common_valid_pixels": common,
            "common_valid_fraction_of_s1": common / int(valid_s1.sum()) if valid_s1.any() else None,
            "label_agreement_fraction": float((labels_s1[valid_common] == labels_s2[valid_common]).mean()) if common else None,
            "s1_class1_pixels_in_common": int(flood_s1.sum()),
            "s2_class1_pixels_in_common": int(flood_s2.sum()),
            "class1_union_pixels": union,
            "class1_iou": float((flood_s1 & flood_s2).sum() / union) if union else None,
            "s1_classes": ";".join(map(str, np.unique(labels_s1).tolist())),
            "s2_classes": ";".join(map(str, np.unique(s2.read(1)).tolist())),
        }
        return result


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    missing = [row["tile_id"] for row in candidates if not all((MASKS / sensor / row["tile_id"]).exists() for sensor in ("S1", "S2"))]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} candidate masks; first: {missing[0]}")
    rows = [audit_pair(row) for row in candidates]
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["ems_code"]].append(row)
    summaries = []
    for activation, members in sorted(groups.items()):
        overlap = [row["common_valid_fraction_of_s1"] or 0 for row in members]
        summaries.append({
            "ems_code": activation,
            "candidate_pairs": len(members),
            "nonzero_overlap": sum(value > 0 for value in overlap),
            "overlap_ge_50pct": sum(value >= 0.5 for value in overlap),
            "overlap_ge_80pct": sum(value >= 0.8 for value in overlap),
            "overlap_ge_90pct": sum(value >= 0.9 for value in overlap),
            "same_floodmap_id": sum(str(row["same_floodmap_id"]).lower() == "true" for row in members),
            "s2_before_s1": sum(float(row["delta_s2_minus_s1_days"]) <= 0 for row in members),
            "class1_disagreement_ge_20pct_with_union": sum(row["class1_iou"] is not None and row["class1_iou"] < 0.8 for row in members),
        })
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    totals = Counter()
    for row in summaries:
        for key, value in row.items():
            if key != "ems_code":
                totals[key] += value
    print(f"Audited {len(rows)} candidate pairs across {len(groups)} EMS activations")
    print(dict(totals))
    print(f"Wrote {OUTPUT} and {SUMMARY}")


if __name__ == "__main__":
    main()
