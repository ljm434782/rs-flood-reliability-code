"""Check whether official EMSR251 MONIT01 flood polygons intersect one holdout chip."""

import csv
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data" / "manifests" / "cems_index.csv"
OUTPUT = ROOT / "data" / "manifests" / "emsr251_holdout_vector_overlap.json"
SAMPLE_ID = "EMSR251_3_32VML_x473405_y6512885"
LABEL = ROOT / "data" / "raw" / "impactmesh" / "samples" / SAMPLE_ID / f"{SAMPLE_ID}_annotation_flood.tif"
VECTOR_MEMBER = "EMSR251_03KRAGERO_DEL_MONIT01_v1_observed_event_a.shp"
AOI_MEMBER = "EMSR251_03KRAGERO_DEL_MONIT01_v1_area_of_interest_a.shp"


def main() -> None:
    with INDEX.open(newline="", encoding="utf-8") as handle:
        source = next(row for row in csv.DictReader(handle) if row["product"] == "03KRAGERO_01DELINEATION_MONIT01_v1_vector")
    archive = ROOT / source["local_path"]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise RuntimeError("CEMS vector archive SHA256 differs from source manifest")
    vectors = gpd.read_file(f"zip://{archive.resolve()}!/{VECTOR_MEMBER}")
    aoi = gpd.read_file(f"zip://{archive.resolve()}!/{AOI_MEMBER}")
    if not vectors.crs or not all(vectors["event_type"] == "5-Flood"):
        raise RuntimeError("Unexpected flood vector CRS or event type")
    if not aoi.crs or len(aoi) != 1:
        raise RuntimeError("Unexpected AOI vector CRS or feature count")
    with rasterio.open(LABEL) as raster:
        values = raster.read(1)
        chip = box(*raster.bounds)
        projected = vectors.to_crs(raster.crs)
        aoi_projected = aoi.to_crs(raster.crs)
        intersections = projected.geometry.intersection(chip)
        result = {
            "sample_id": SAMPLE_ID,
            "sample_label_path": LABEL.relative_to(ROOT).as_posix(),
            "sample_label_sha256": hashlib.sha256(LABEL.read_bytes()).hexdigest(),
            "sample_label_crs": str(raster.crs),
            "sample_bounds": list(raster.bounds),
            "sample_flood_class_2_pixels": int(np.count_nonzero(values == 2)),
            "cems_product": source["product"],
            "cems_vector_source_url": source["source_url"],
            "cems_vector_sha256": digest,
            "cems_vector_member": VECTOR_MEMBER,
            "cems_aoi_member": AOI_MEMBER,
            "cems_vector_crs": str(vectors.crs),
            "chip_inside_aoi_fraction": float(aoi_projected.geometry.intersection(chip).area.sum() / chip.area),
            "flood_polygon_count": len(vectors),
            "intersecting_flood_polygon_count": int(projected.geometry.intersects(chip).sum()),
            "flood_intersection_area_m2": float(intersections.area.sum()),
            "nearest_flood_polygon_distance_m": float(projected.geometry.distance(chip).min()),
            "label_product_link": "unverified",
        }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Flood intersections: {result['intersecting_flood_polygon_count']}; nearest {result['nearest_flood_polygon_distance_m']:.1f} m")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
