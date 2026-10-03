"""Verify masks and test spatial pairing in four no-shared-name STURM activations."""

import csv
import hashlib
import json
import zlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import rasterio
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.ops import transform as project


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sturm"
OUT = ROOT / "data" / "manifests"
ACTIVATIONS = ("EMSR424", "EMSR444", "EMSR554", "EMSR567")
FOLDERS = {"S1": "Sentinel1", "S2": "Sentinel2"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with (OUT / "sturm_zip_index.csv").open(newline="", encoding="utf-8") as handle:
        source = {row["path"]: row for row in csv.DictReader(handle)}
    metadata = {}
    full_metadata = {}
    for sensor, folder in FOLDERS.items():
        with (RAW / "metadata" / f"{folder}_metadata.csv").open(newline="", encoding="utf-8") as handle:
            full_metadata[sensor] = list(csv.DictReader(handle))
        metadata[sensor] = {row["tile_id"]: row for row in full_metadata[sensor]
                            if row["ems_code"] in ACTIVATIONS}
    both = {row["ems_code"] for row in full_metadata["S1"]} & {row["ems_code"] for row in full_metadata["S2"]}
    s2_ids = {row["tile_id"] for row in full_metadata["S2"]}
    shared_tiles = {row["ems_code"] for row in full_metadata["S1"]
                    if row["tile_id"] in s2_ids}
    if both - shared_tiles != set(ACTIVATIONS):
        raise RuntimeError("Activations without same-name tiles changed")
    member_rows = []
    footprints = defaultdict(lambda: defaultdict(list))
    transformer_cache = {}
    for sensor, folder in FOLDERS.items():
        directory = RAW / "unmatched_activation_masks" / sensor
        actual = {path.name for path in directory.glob("*.tif")}
        if actual != set(metadata[sensor]):
            raise RuntimeError(f"Local {sensor} masks differ from original metadata")
        for tile_id, item in sorted(metadata[sensor].items()):
            path = directory / tile_id
            data = path.read_bytes()
            archive_name = f"Dataset/{folder}/Floodmaps/{tile_id}"
            record = source[archive_name]
            crc = f"{zlib.crc32(data) & 0xffffffff:08x}"
            if len(data) != int(record["size_bytes"]) or crc != record["crc32"]:
                raise RuntimeError(f"Mask differs from ZIP index: {path}")
            with rasterio.open(path) as raster:
                if (raster.shape != (128, 128) or raster.res != (10.0, 10.0) or
                        raster.nodata != 99 or raster.count != 1 or
                        raster.crs.to_epsg() != int(item["epsg_code"])):
                    raise RuntimeError(f"Unexpected source mask grid: {path}")
                affine = raster.transform
                corners = [affine * (x, y) for x, y in ((0, 0), (128, 0), (128, 128), (0, 128))]
                native = Polygon(corners)
                crs = raster.crs.to_string()
                if crs not in transformer_cache:
                    transformer_cache[crs] = Transformer.from_crs(crs, "EPSG:6933", always_xy=True).transform
                geometry = project(transformer_cache[crs], native)
                bounds = [float(value) for value in raster.bounds]
            row = {
                "sensor": sensor, "activation": item["ems_code"], "aoi_code": item["aoi_code"],
                "tile_id": tile_id, "source_zip_member": archive_name,
                "source_zip_crc32": crc, "source_member_bytes": len(data),
                "local_sha256": sha(path), "crs": crs,
                "left": bounds[0], "bottom": bounds[1], "right": bounds[2], "top": bounds[3],
            }
            member_rows.append(row)
            footprints[item["ems_code"]][sensor].append((item, geometry))
    member_path = OUT / "sturm_unmatched_activation_mask_manifest.csv"
    with member_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(member_rows[0]))
        writer.writeheader()
        writer.writerows(member_rows)

    pair_rows = []
    activation_rows = []
    for code in ACTIVATIONS:
        s1 = footprints[code]["S1"]
        s2 = footprints[code]["S2"]
        same_aoi = 0
        positive = 0
        minimum_distance_m = float("inf")
        for first, first_geom in s1:
            for second, second_geom in s2:
                same_aoi += int(first["aoi_code"] == second["aoi_code"])
                minimum_distance_m = min(minimum_distance_m, first_geom.distance(second_geom))
                if not first_geom.intersects(second_geom):
                    continue
                area = first_geom.intersection(second_geom).area
                if area <= 0:
                    continue
                positive += 1
                first_time = datetime.fromisoformat(first["sentinel_date"])
                second_time = datetime.fromisoformat(second["sentinel_timestamp"])
                pair_rows.append({
                    "activation": code, "s1_tile_id": first["tile_id"],
                    "s2_tile_id": second["tile_id"],
                    "s1_aoi": first["aoi_code"], "s2_aoi": second["aoi_code"],
                    "intersection_m2_equal_area": area,
                    "s1_reference_overlap_fraction": area / first_geom.area,
                    "s2_minus_s1_hours": (second_time - first_time).total_seconds() / 3600,
                    "same_floodmap_id": int(first["floodmap_id"] == second["floodmap_id"]),
                })
        activation_rows.append({
            "activation": code, "s1_masks": len(s1), "s2_masks": len(s2),
            "s1_aois": dict(Counter(item["aoi_code"] for item, _ in s1)),
            "s2_aois": dict(Counter(item["aoi_code"] for item, _ in s2)),
            "same_aoi_cartesian_pairs": same_aoi,
            "positive_area_spatial_pairs": positive,
            "minimum_footprint_distance_km_equal_area": minimum_distance_m / 1000,
        })
    pairs_path = OUT / "sturm_unmatched_spatial_pairs.csv"
    fields = ["activation", "s1_tile_id", "s2_tile_id", "s1_aoi", "s2_aoi",
              "intersection_m2_equal_area", "s1_reference_overlap_fraction",
              "s2_minus_s1_hours", "same_floodmap_id"]
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_rows)
    result = {
        "scope": "four dual-modality STURM activations without same-name tiles; all published S1/S2 floodmap footprints",
        "source_zip_index_sha256": sha(OUT / "sturm_zip_index.csv"),
        "source_metadata_sha256": {sensor: sha(RAW / "metadata" / f"{folder}_metadata.csv")
                                   for sensor, folder in FOLDERS.items()},
        "member_manifest_sha256": sha(member_path),
        "spatial_pairs_sha256": sha(pairs_path),
        "checked_masks": len(member_rows),
        "positive_area_spatial_pairs": len(pair_rows),
        "activations": activation_rows,
        "limitations": [
            "Mask footprints establish geometric overlap only; they do not establish imagery alignment, common valid pixels, or label agreement.",
            "ZIP member CRC and local SHA256 are checked; the full publisher Dataset.zip MD5 is not verified.",
            "Different AOI codes are not assumed spatially disjoint; every cross-modality footprint pair is tested.",
        ],
    }
    output = OUT / "sturm_unmatched_activation_audit.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(activation_rows, ensure_ascii=False))
    print(output)


if __name__ == "__main__":
    main()
