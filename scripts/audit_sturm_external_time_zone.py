"""Audit STURM masks near Sen1 hand-test dates and adjacent UTM zones."""

import csv
import hashlib
import json
import zipfile
import zlib
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

from index_sturm_zip import RangeReader


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "manifests"
MASKS = RAW / "sturm" / "external_time_zone_masks"
WINDOW_DAYS = 30


def overlaps(left, right):
    return max(left[0], right[0]) < min(left[2], right[2]) and max(left[1], right[1]) < min(left[3], right[3])


def candidate_epsgs(bounds):
    west, south, east, north = bounds
    if east - west >= 180:
        raise ValueError("Unexpected antimeridian-spanning external scene")
    zones = set()
    for longitude in (west, east):
        central = min(60, max(1, int((longitude + 180) // 6) + 1))
        zones.update(range(max(1, central - 1), min(60, central + 1) + 1))
    prefixes = set()
    if north >= 0:
        prefixes.add(326)
    if south <= 0:
        prefixes.add(327)
    return {prefix * 100 + zone for prefix in prefixes for zone in zones}


def main():
    geojson = json.loads((RAW / "sen1floods11" / "Sen1Floods11_Metadata.geojson").read_text(encoding="utf-8"))
    dates = {
        feature["properties"]["location"]: datetime.strptime(feature["properties"]["s1_date"], "%Y/%m/%d")
        for feature in geojson["features"]
    }
    scenes = []
    for path in sorted((RAW / "sen1floods11" / "data" / "flood_events" / "HandLabeled" / "LabelHand").glob("*.tif")):
        event = path.stem.split("_", 1)[0]
        metadata_event = "Cambodia" if event == "Mekong" else event
        with rasterio.open(path) as source:
            bounds = transform_bounds(source.crs, "EPSG:4326", *source.bounds, densify_pts=21)
        scenes.append({
            "scene": path.stem.removesuffix("_LabelHand"),
            "event": event,
            "event_date": dates[metadata_event],
            "bounds": bounds,
            "epsgs": candidate_epsgs(bounds),
        })
    if len(scenes) != 90:
        raise RuntimeError(f"Expected 90 external scenes, found {len(scenes)}")

    candidates = {}
    for sensor, folder, time_field in (
        ("S1", "Sentinel1", "sentinel_date"),
        ("S2", "Sentinel2", "sentinel_timestamp"),
    ):
        with (RAW / "sturm" / "metadata" / f"{folder}_metadata.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                timestamp = datetime.fromisoformat(row[time_field])
                epsg = int(row["epsg_code"])
                matched = [
                    scene for scene in scenes
                    if epsg in scene["epsgs"]
                    and abs(timestamp - scene["event_date"]) <= timedelta(days=WINDOW_DAYS)
                ]
                if not matched:
                    continue
                tile_id = row["tile_id"]
                if Path(tile_id).name != tile_id or not tile_id.endswith(".tif"):
                    raise ValueError(f"Unsafe tile ID: {tile_id}")
                key = (sensor, tile_id)
                entry = candidates.setdefault(key, {
                    "sensor": sensor, "folder": folder, "tile_id": tile_id,
                    "ems_code": row["ems_code"], "country": row["country"],
                    "epsg_code": epsg, "timestamp": timestamp.isoformat(),
                    "scenes": set(),
                })
                entry["scenes"].update(scene["scene"] for scene in matched)

    record = json.loads((OUT / "sturm_zenodo_record.json").read_text(encoding="utf-8"))
    asset = next(item for item in record["files"] if item["key"] == "Dataset.zip")
    with zipfile.ZipFile(RangeReader(asset["links"]["self"], asset["size"])) as archive:
        targets = []
        for entry in candidates.values():
            member = f"Dataset/{entry['folder']}/Floodmaps/{entry['tile_id']}"
            targets.append((archive.getinfo(member), MASKS / entry["sensor"] / entry["tile_id"], entry))
        for info, target, entry in sorted(targets, key=lambda item: item[0].header_offset):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                content = target.read_bytes()
                if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
                    raise RuntimeError(f"Existing mask fails source ZIP CRC/size: {target}")
            else:
                content = archive.read(info.filename)
                target.write_bytes(content)
            entry["source_crc32"] = f"{info.CRC:08x}"
            entry["sha256"] = hashlib.sha256(content).hexdigest()
            with rasterio.open(target) as source:
                entry["bounds"] = transform_bounds(source.crs, "EPSG:4326", *source.bounds, densify_pts=21)

    by_scene = defaultdict(list)
    for entry in candidates.values():
        for scene_id in entry["scenes"]:
            by_scene[scene_id].append(entry)
    per_scene = []
    for scene in scenes:
        nearby = by_scene[scene["scene"]]
        spatial = [entry for entry in nearby if overlaps(entry["bounds"], scene["bounds"])]
        per_scene.append({
            "scene": scene["scene"], "event": scene["event"],
            "time_zone_candidate_masks": len(nearby),
            "spatial_bbox_overlap_masks": len(spatial),
            "spatial_bbox_overlap_activation_ids": sorted({entry["ems_code"] for entry in spatial}),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "sturm_external_time_zone_masks.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fields = ("sensor", "tile_id", "ems_code", "country", "epsg_code", "timestamp", "scene_ids", "source_crc32", "sha256", "west", "south", "east", "north")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in sorted(candidates.values(), key=lambda value: (value["sensor"], value["tile_id"])):
            writer.writerow({
                **{key: entry[key] for key in fields[:6]},
                "scene_ids": ";".join(sorted(entry["scenes"])),
                "source_crc32": entry["source_crc32"], "sha256": entry["sha256"],
                **dict(zip(fields[-4:], entry["bounds"])),
            })
    summary = {
        "window_days": WINDOW_DAYS,
        "zone_rule": "External scene longitude UTM zone plus one adjacent zone on either side; scene hemisphere",
        "external_scenes": len(scenes),
        "candidate_masks": len(candidates),
        "candidate_activations": sorted({entry["ems_code"] for entry in candidates.values()}),
        "scenes_with_candidate_masks": sum(row["time_zone_candidate_masks"] > 0 for row in per_scene),
        "scenes_with_spatial_bbox_overlap": sum(row["spatial_bbox_overlap_masks"] > 0 for row in per_scene),
        "per_scene": per_scene,
        "source": asset["links"]["self"],
    }
    (OUT / "sturm_external_time_zone_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("candidate_masks", "candidate_activations", "scenes_with_candidate_masks", "scenes_with_spatial_bbox_overlap")}, indent=2))


if __name__ == "__main__":
    main()
