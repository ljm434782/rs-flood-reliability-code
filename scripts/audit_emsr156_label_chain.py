"""Compare one fixed train label with both official EMSR156 delineation products."""

import csv
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
SAMPLE_ID = "EMSR156_1_29UNV_x545285_y5951505"
LABEL = ROOT / "data" / "raw" / "impactmesh" / "samples" / SAMPLE_ID / f"{SAMPLE_ID}_annotation_flood.tif"
OUTPUT = MANIFESTS / "emsr156_train_label_chain.json"


def verified_sources() -> dict[str, dict[str, str]]:
    with (MANIFESTS / "cems_index.csv").open(newline="", encoding="utf-8") as handle:
        sources = {row["product"]: row for row in csv.DictReader(handle) if row["activation"] == "EMSR156"}
    if len(sources) != 4:
        raise RuntimeError("Expected both EMSR156 AOI maps and vector packages")
    for source in sources.values():
        path = ROOT / source["local_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError(f"CEMS source hash mismatch: {path}")
    return sources


def read_layer(path: Path, suffix: str) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f"Expected exactly one {suffix} layer in {path}")
    return gpd.read_file(f"zip://{path.resolve()}!/{names[0]}")


def main() -> None:
    sources = verified_sources()
    audit = json.loads((MANIFESTS / f"impactmesh_sample_audit_{SAMPLE_ID}.json").read_text(encoding="utf-8"))
    event_s1 = datetime.fromisoformat(audit["S1RTC"]["timestamps_utc"][2].replace("Z", "+00:00"))
    with (MANIFESTS / "impactmesh_sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        label_entries = [row for row in csv.DictReader(handle) if row["sample_id"] == SAMPLE_ID and row["split"] == "train" and row["modality"] == "MASK"]
    if len(label_entries) != 1:
        raise RuntimeError("Expected one train MASK source record")
    label_digest = hashlib.sha256(LABEL.read_bytes()).hexdigest()
    if label_digest != label_entries[0]["sha256"] or int(label_entries[0]["size_bytes"]) != LABEL.stat().st_size:
        raise RuntimeError("ImpactMesh label differs from saved source manifest")
    with rasterio.open(LABEL) as raster:
        label = raster.read(1)
        unique, counts = np.unique(label, return_counts=True)
        class_counts = {str(key): int(value) for key, value in zip(unique.tolist(), counts.tolist())}
        if class_counts != audit["label"]["class_counts"]:
            raise RuntimeError("Current label class counts differ from saved sample audit")
        chip = box(*raster.bounds)
        result = {
            "sample_id": SAMPLE_ID,
            "split": "train",
            "label_path": LABEL.relative_to(ROOT).as_posix(),
            "label_sha256": label_digest,
            "label_crs": str(raster.crs),
            "label_bounds": list(raster.bounds),
            "label_class_counts": class_counts,
            "event_s1_utc": event_s1.isoformat().replace("+00:00", "Z"),
            "products": [],
            "label_product_link": "unverified; spatial and temporal comparison only",
        }
        for code in ("01CASTELPLUNKET", "02LOUGHFUNSHINAGH"):
            vector = sources[f"{code}_DELINEATION_OVERVIEW_v1_vector"]
            map_source = sources[f"{code}_DELINEATION_OVERVIEW_v1_map"]
            archive_path = ROOT / vector["local_path"]
            aoi = read_layer(archive_path, "area_of_interest.shp")
            flood = read_layer(archive_path, "crisis_information_poly.shp")
            if not aoi.crs or not flood.crs or len(aoi) == 0 or len(flood) == 0:
                raise RuntimeError(f"Invalid CEMS vector layers: {code}")
            if not all(flood["interpret"] == "Flooded Area"):
                raise RuntimeError(f"Unexpected CEMS crisis classes: {code}")
            aoi_projected = aoi.to_crs(raster.crs)
            flood_projected = flood.to_crs(raster.crs)
            aoi_inside = float(aoi_projected.geometry.intersection(chip).area.sum() / chip.area)
            flood_mask = rasterize(
                ((geometry, 1) for geometry in flood_projected.geometry if not geometry.is_empty),
                out_shape=label.shape, transform=raster.transform, fill=0, dtype="uint8", all_touched=False,
            ).astype(bool)
            valid = label != -1
            labeled_flood = label == 2
            tp = int(np.count_nonzero(flood_mask & labeled_flood & valid))
            fp = int(np.count_nonzero(~flood_mask & labeled_flood & valid))
            fn = int(np.count_nonzero(flood_mask & ~labeled_flood & valid))
            union = tp + fp + fn
            reference_time = datetime.fromisoformat(map_source["source_image_time_utc"].replace("Z", "+00:00"))
            result["products"].append({
                "code": code,
                "map_source_url": map_source["source_url"],
                "map_sha256": map_source["sha256"],
                "vector_source_url": vector["source_url"],
                "vector_sha256": vector["sha256"],
                "reference_image": "RADARSAT-2; map legend and Data Sources, manually checked",
                "reference_image_utc": map_source["source_image_time_utc"],
                "reference_minus_event_s1_minutes": (reference_time - event_s1).total_seconds() / 60,
                "aoi_crs": str(aoi.crs),
                "flood_crs": str(flood.crs),
                "aoi_feature_count": len(aoi),
                "flood_polygon_count": len(flood),
                "chip_inside_aoi_fraction": aoi_inside,
                "flood_intersection_area_m2": float(flood_projected.geometry.intersection(chip).area.sum()),
                "rasterized_cems_flood_pixels_in_chip": int(np.count_nonzero(flood_mask)),
                "rasterized_cems_flood_pixels_on_label_class_1": int(np.count_nonzero(flood_mask & (label == 1))),
                "class_2_vs_cems_flood_tp": tp,
                "class_2_vs_cems_flood_fp": fp,
                "class_2_vs_cems_flood_fn": fn,
                "class_2_vs_cems_flood_iou": tp / union if union else None,
            })
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for item in result["products"]:
        print(f"{item['code']}: AOI fraction={item['chip_inside_aoi_fraction']:.4f}; "
              f"CEMS flood pixels={item['rasterized_cems_flood_pixels_in_chip']}; "
              f"label flood IoU={item['class_2_vs_cems_flood_iou']}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
