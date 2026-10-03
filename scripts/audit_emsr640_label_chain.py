"""Compare fixed train rank 3 with all three CEMS EMSR640 AOI02 delineations."""

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
SAMPLE_ID = "EMSR640_2_30VWJ_x547805_y6332575"
LABEL = ROOT / "data" / "raw" / "impactmesh" / "samples" / SAMPLE_ID / f"{SAMPLE_ID}_annotation_flood.tif"
OUTPUT = MANIFESTS / "emsr640_train_label_chain.json"


def source_rows() -> dict[str, dict[str, str]]:
    with (MANIFESTS / "cems_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = {row["product"]: row for row in csv.DictReader(handle) if row["activation"] == "EMSR640"}
    if len(rows) != 9:
        raise RuntimeError("Expected four AOI vectors and three AOI02 map/vector pairs")
    for row in rows.values():
        path = ROOT / row["local_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"CEMS file hash mismatch: {path}")
    return rows


def layer(path: Path, name_part: str) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".shp") and name_part in name]
    if len(members) != 1:
        raise RuntimeError(f"Expected one {name_part} layer in {path}")
    return gpd.read_file(f"zip://{path.resolve()}!/{members[0]}")


def raster_mask(geometries, raster: rasterio.io.DatasetReader) -> np.ndarray:
    return rasterize(
        ((geometry, 1) for geometry in geometries if not geometry.is_empty),
        out_shape=raster.shape, transform=raster.transform, fill=0, dtype="uint8", all_touched=False,
    ).astype(bool)


def comparison(mask: np.ndarray, label: np.ndarray) -> dict[str, float | int | None]:
    valid = label != -1
    target = label == 2
    tp = int(np.count_nonzero(mask & target & valid))
    fp = int(np.count_nonzero(~mask & target & valid))
    fn = int(np.count_nonzero(mask & ~target & valid))
    union = tp + fp + fn
    return {
        "cems_mask_pixels": int(np.count_nonzero(mask)),
        "tp": tp, "label_only_fp": fp, "cems_only_fn": fn,
        "cems_only_on_label_class_1": int(np.count_nonzero(mask & (label == 1))),
        "flood_iou": tp / union if union else None,
    }


def main() -> None:
    sources = source_rows()
    with (MANIFESTS / "impactmesh_sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        entries = [row for row in csv.DictReader(handle) if row["sample_id"] == SAMPLE_ID and row["split"] == "train" and row["modality"] == "MASK"]
    if len(entries) != 1:
        raise RuntimeError("Expected one train MASK manifest row")
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
            raise RuntimeError("Current label counts differ from saved audit")
        chip = box(*raster.bounds)
        result = {
            "sample_id": SAMPLE_ID,
            "split": "train",
            "label_sha256": label_digest,
            "label_crs": str(raster.crs),
            "label_bounds": list(raster.bounds),
            "label_class_counts": class_counts,
            "event_s1_utc": event_s1.isoformat().replace("+00:00", "Z"),
            "aoi_localization": [],
            "aoi02_products": [],
            "label_product_link": "unverified; compare spatial footprints and reference-image times",
        }
        for code, version in (("AOI01", "v1"), ("AOI02", "v3"), ("AOI03", "v1"), ("AOI04", "v2")):
            product = f"{code}_DEL_PRODUCT_r1_RTP01_{version}_vector"
            source = sources[product]
            aoi = layer(ROOT / source["local_path"], "areaOfInterestA").to_crs(raster.crs)
            result["aoi_localization"].append({
                "aoi": code,
                "source_sha256": source["sha256"],
                "chip_inside_aoi_fraction": float(aoi.geometry.intersection(chip).area.sum() / chip.area),
            })
        fractions = {item["aoi"]: item["chip_inside_aoi_fraction"] for item in result["aoi_localization"]}
        if fractions["AOI02"] < 0.999999 or any(fractions[code] > 1e-9 for code in ("AOI01", "AOI03", "AOI04")):
            raise RuntimeError("Sample does not belong exclusively to AOI02")
        product_masks = {}
        for name in ("PRODUCT", "MONIT01", "MONIT02"):
            base = f"AOI02_DEL_{name}_r1_RTP01_v3"
            vector = sources[f"{base}_vector"]
            map_source = sources[f"{base}_map"]
            observed = layer(ROOT / vector["local_path"], "observedEventA").to_crs(raster.crs)
            if not all(observed["event_type"] == "5-Flood"):
                raise RuntimeError(f"Unexpected event type in {name}")
            if not set(observed["notation"]) <= {"Flooded area", "Previous flooded area"}:
                raise RuntimeError(f"Unexpected observed-event notation in {name}")
            timestamp = datetime.fromisoformat(map_source["source_image_time_utc"].replace("Z", "+00:00"))
            item = {
                "product": name,
                "map_source_url": map_source["source_url"],
                "map_sha256": map_source["sha256"],
                "vector_source_url": vector["source_url"],
                "vector_sha256": vector["sha256"],
                "latest_reference_image_utc": map_source["source_image_time_utc"],
                "latest_reference_minus_event_s1_hours": (timestamp - event_s1).total_seconds() / 3600,
                "observed_event_polygon_count": len(observed),
                "notations": observed["notation"].value_counts().to_dict(),
                "comparisons": {},
            }
            for label_name, notations in (
                    ("current", {"Flooded area"}),
                    ("previous", {"Previous flooded area"}),
                    ("current_plus_previous", {"Flooded area", "Previous flooded area"})):
                subset = observed[observed["notation"].isin(notations)]
                mask = raster_mask(subset.geometry, raster)
                product_masks[(name, label_name)] = mask
                item["comparisons"][label_name] = comparison(mask, label)
            result["aoi02_products"].append(item)
        result["carried_forward_footprints"] = {
            "product_current_vs_monit01_previous_differing_pixels": int(np.count_nonzero(
                product_masks[("PRODUCT", "current")] ^ product_masks[("MONIT01", "previous")]
            )),
            "monit01_current_vs_monit02_previous_differing_pixels": int(np.count_nonzero(
                product_masks[("MONIT01", "current")] ^ product_masks[("MONIT02", "previous")]
            )),
        }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for item in result["aoi02_products"]:
        print(item["product"], item["latest_reference_minus_event_s1_hours"],
              {name: value["flood_iou"] for name, value in item["comparisons"].items()})
    print(OUTPUT)


if __name__ == "__main__":
    main()
