"""Summarize spatial alignment and label values in selectively fetched STURM samples."""

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "raw" / "sturm" / "samples"
OUTPUT = ROOT / "data" / "manifests" / "sturm_sample_grid_audit.csv"


def values(path: Path) -> str:
    with rasterio.open(path) as source:
        unique = np.unique(source.read(1))
        return ";".join(map(str, unique.tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    rows = []
    for folder in sorted(SAMPLES.iterdir()):
        if not folder.is_dir():
            continue
        if not all((folder / filename).exists() for filename in ("S1.tif", "S2.tif", "S1_floodmap.tif", "S2_floodmap.tif")):
            continue
        with rasterio.open(folder / "S1.tif") as s1, rasterio.open(folder / "S2.tif") as s2:
            if s1.crs != s2.crs:
                overlap_fraction = None
            else:
                left = max(s1.bounds.left, s2.bounds.left)
                right = min(s1.bounds.right, s2.bounds.right)
                bottom = max(s1.bounds.bottom, s2.bounds.bottom)
                top = min(s1.bounds.top, s2.bounds.top)
                area = max(0, right - left) * max(0, top - bottom)
                overlap_fraction = area / ((s1.bounds.right - s1.bounds.left) * (s1.bounds.top - s1.bounds.bottom))
            with rasterio.open(folder / "S1_floodmap.tif") as s1_mask, rasterio.open(folder / "S2_floodmap.tif") as s2_mask:
                s1_labels = s1_mask.read(1)
                aligned_s2 = np.full(s1_labels.shape, 255, dtype=np.int32)
                source_s2 = s2_mask.read(1).astype(np.int32)
                reproject(
                    source=source_s2,
                    destination=aligned_s2,
                    src_transform=s2_mask.transform,
                    src_crs=s2_mask.crs,
                    src_nodata=s2_mask.nodata,
                    dst_transform=s1_mask.transform,
                    dst_crs=s1_mask.crs,
                    dst_nodata=255,
                    resampling=Resampling.nearest,
                )
                valid = (s1_labels != s1_mask.nodata) & (aligned_s2 != 255)
                common_pixels = int(valid.sum())
                agreement = float((s1_labels[valid] == aligned_s2[valid]).mean()) if common_pixels else None
                s1_flood = (s1_labels == 1) & valid
                s2_flood = (aligned_s2 == 1) & valid
                union = int((s1_flood | s2_flood).sum())
                flood_iou = float((s1_flood & s2_flood).sum() / union) if union else None
            rows.append({
                "tile_id": folder.name + ".tif",
                "s1_crs": str(s1.crs),
                "s2_crs": str(s2.crs),
                "s1_bands": s1.count,
                "s2_bands": s2.count,
                "s1_nodata": s1.nodata,
                "s2_nodata": s2.nodata,
                "dx_m": s2.bounds.left - s1.bounds.left,
                "dy_m": s2.bounds.top - s1.bounds.top,
                "same_transform": s1.transform == s2.transform,
                "overlap_fraction_of_s1": overlap_fraction,
                "common_valid_label_pixels": common_pixels,
                "label_agreement_fraction": agreement,
                "flood_class_iou": flood_iou,
                "s1_mask_values": values(folder / "S1_floodmap.tif"),
                "s2_mask_values": values(folder / "S2_floodmap.tif"),
            })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audited {len(rows)} sampled tile IDs; wrote {OUTPUT}")
    for row in rows:
        overlap = "different CRS" if row["overlap_fraction_of_s1"] is None else f"{row['overlap_fraction_of_s1']:.1%}"
        agreement = "n/a" if row["label_agreement_fraction"] is None else f"{row['label_agreement_fraction']:.1%}"
        print(f"{row['tile_id']}: dy={row['dy_m']:.2f} m, overlap={overlap}, label_agreement={agreement}")


if __name__ == "__main__":
    main()
