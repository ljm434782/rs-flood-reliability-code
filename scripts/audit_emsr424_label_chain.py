"""Compare fixed train rank 4 with three official EMSR424 AOI01 products."""

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
SAMPLE_ID = "EMSR424_1_38LQH_x786235_y8282875"
LABEL = ROOT / "data" / "raw" / "impactmesh" / "samples" / SAMPLE_ID / f"{SAMPLE_ID}_annotation_flood.tif"
OUTPUT = MANIFESTS / "emsr424_train_label_chain.json"
PRODUCTS = {
    "first_estimate": (
        "AOI01_FEP_PRODUCT_r1_RTP01_v2",
        "AOI01_FEP_PRODUCT_r1_VECTORS_v1_vector",
    ),
    "delineation": (
        "AOI01_DEL_PRODUCT_r1_RTP01_v2",
        "AOI01_DEL_PRODUCT_r1_RTP01_v2_vector",
    ),
    "monitoring_01": (
        "AOI01_DEL_MONIT01_r1_RTP01_v1",
        "AOI01_DEL_MONIT01_r1_RTP01_v1_vector",
    ),
}


def source_rows() -> dict[str, dict[str, str]]:
    with (MANIFESTS / "cems_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["product"]: row for row in csv.DictReader(handle) if row["activation"] == "EMSR424"}
    expected = {name for pair in PRODUCTS.values() for name in pair}
    if set(rows) != expected:
        raise RuntimeError(f"Unexpected EMSR424 products: {sorted(set(rows) ^ expected)}")
    for row in rows.values():
        path = ROOT / row["local_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"CEMS source hash mismatch: {path}")
        if row["status"] != "source-map-checked" or not row["source_image_time_utc"]:
            raise RuntimeError(f"Missing manually verified source time: {row['product']}")
    return rows


def layer(path: Path, name_part: str) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".shp") and name_part in name]
    if len(members) != 1:
        raise RuntimeError(f"Expected one {name_part} layer in {path}")
    return gpd.read_file(f"zip://{path.resolve()}!/{members[0]}")


def main() -> None:
    sources = source_rows()
    with (MANIFESTS / "impactmesh_sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        entries = [row for row in csv.DictReader(handle) if row["sample_id"] == SAMPLE_ID and row["split"] == "train" and row["modality"] == "MASK"]
    if len(entries) != 1:
        raise RuntimeError("Expected one train MASK source manifest row")
    label_digest = hashlib.sha256(LABEL.read_bytes()).hexdigest()
    if label_digest != entries[0]["sha256"] or LABEL.stat().st_size != int(entries[0]["size_bytes"]):
        raise RuntimeError("ImpactMesh label differs from source manifest")
    audit = json.loads((MANIFESTS / f"impactmesh_sample_audit_{SAMPLE_ID}.json").read_text(encoding="utf-8"))
    event_s1 = datetime.fromisoformat(audit["S1RTC"]["timestamps_utc"][2].replace("Z", "+00:00"))

    with rasterio.open(LABEL) as raster:
        label = raster.read(1)
        unique, counts = np.unique(label, return_counts=True)
        class_counts = {str(key): int(value) for key, value in zip(unique.tolist(), counts.tolist())}
        if class_counts != audit["label"]["class_counts"]:
            raise RuntimeError("Current label counts differ from saved sample audit")
        chip = box(*raster.bounds)
        valid = label != -1
        target = label == 2
        result = {
            "sample_id": SAMPLE_ID,
            "split": "train",
            "selection": "fixed train-only temporal-audit frame rank 4",
            "label_sha256": label_digest,
            "label_crs": str(raster.crs),
            "label_bounds": list(raster.bounds),
            "label_class_counts": class_counts,
            "event_s1_utc": event_s1.isoformat().replace("+00:00", "Z"),
            "products": [],
            "product_version_link": "unverified; spatial match does not identify the exact CEMS file used by ImpactMesh",
        }
        masks = {}
        source_masks = {}
        for stage, (map_name, vector_name) in PRODUCTS.items():
            map_row, vector_row = sources[map_name], sources[vector_name]
            vector_path = ROOT / vector_row["local_path"]
            aoi = layer(vector_path, "areaOfInterestA").to_crs(raster.crs)
            observed = layer(vector_path, "observedEventA").to_crs(raster.crs)
            if "notation" not in observed:
                raise RuntimeError(f"Missing observedEventA notation in {vector_name}")
            flooded = observed[observed["notation"] == "Flooded area"]
            def make_mask(frame: gpd.GeoDataFrame) -> np.ndarray:
                return rasterize(
                    ((geometry, 1) for geometry in frame.geometry if not geometry.is_empty),
                    out_shape=raster.shape,
                    transform=raster.transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                ).astype(bool)
            mask = make_mask(flooded)
            masks[stage] = mask
            tp = int(np.count_nonzero(mask & target & valid))
            label_only = int(np.count_nonzero(~mask & target & valid))
            cems_only = int(np.count_nonzero(mask & ~target & valid))
            union = tp + label_only + cems_only
            # A monitoring map can retain an earlier flood polygon. Its newest
            # image time is not necessarily the reference time of that polygon.
            latest_map_source_time = datetime.fromisoformat(map_row["source_image_time_utc"].replace("Z", "+00:00"))
            source_slices = []
            if "dmg_src_id" in flooded:
                for source_id, frame in flooded.groupby("dmg_src_id"):
                    source_mask = make_mask(frame)
                    source_masks[(stage, int(source_id))] = source_mask
                    source_tp = int(np.count_nonzero(source_mask & target & valid))
                    source_label_only = int(np.count_nonzero(~source_mask & target & valid))
                    source_cems_only = int(np.count_nonzero(source_mask & ~target & valid))
                    source_union = source_tp + source_label_only + source_cems_only
                    source_slices.append({
                        "dmg_src_id": int(source_id),
                        "flood_pixels": int(np.count_nonzero(source_mask)),
                        "flood_iou": source_tp / source_union if source_union else None,
                        "label_only": source_label_only,
                        "cems_only": source_cems_only,
                    })
            result["products"].append({
                "stage": stage,
                "map_product": map_name,
                "map_sha256": map_row["sha256"],
                "map_source_url": map_row["source_url"],
                "vector_product": vector_name,
                "vector_sha256": vector_row["sha256"],
                "map_latest_source_image_time_utc": map_row["source_image_time_utc"],
                "map_latest_source_minus_event_s1_hours": (latest_map_source_time - event_s1).total_seconds() / 3600,
                "aoi_chip_area_fraction": aoi.geometry.union_all().intersection(chip).area / chip.area,
                "observed_event_notations": {str(key): int(value) for key, value in observed["notation"].value_counts(dropna=False).items()},
                "flood_polygons_intersecting_chip": int(np.count_nonzero(flooded.geometry.intersects(chip))),
                "cems_flood_pixels": int(np.count_nonzero(mask)),
                "tp": tp,
                "label_only": label_only,
                "cems_only": cems_only,
                "cems_only_on_label_class_1": int(np.count_nonzero(mask & (label == 1))),
                "flood_iou": tp / union if union else None,
                "source_id_slices": source_slices,
            })
        result["flood_mask_pairwise_differing_pixels"] = {
            "first_estimate_vs_delineation": int(np.count_nonzero(masks["first_estimate"] != masks["delineation"])),
            "delineation_vs_monitoring_01": int(np.count_nonzero(masks["delineation"] != masks["monitoring_01"])),
        }
        result["monitoring_01_source_comparison"] = {
            "source_3_vs_delineation_differing_pixels": int(np.count_nonzero(source_masks[("monitoring_01", 3)] != masks["delineation"])),
            "source_5_outside_source_3_pixels": int(np.count_nonzero(source_masks[("monitoring_01", 5)] & ~source_masks[("monitoring_01", 3)])),
        }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    for row in result["products"]:
        print(row["stage"], row["map_latest_source_image_time_utc"], f"IoU={row['flood_iou']:.4f}")


if __name__ == "__main__":
    main()
