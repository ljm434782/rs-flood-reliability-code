"""Check Sen1Floods11 hand-test scenes against STURM S1/S2 country, date, and footprint."""

import csv
import json
from datetime import datetime
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds


ROOT = Path(__file__).resolve().parents[1]
SEN_ROOT = ROOT / "data" / "raw" / "sen1floods11"
STURM_ROOT = ROOT / "data" / "raw" / "sturm"
OUTPUT = ROOT / "data" / "manifests" / "sen1_sturm_event_overlap_audit.csv"
EVENT_COUNTRY = {"Mekong": "Cambodia"}


def intersects(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def main() -> None:
    geojson = json.loads((SEN_ROOT / "Sen1Floods11_Metadata.geojson").read_text(encoding="utf-8"))
    events = {feature["properties"]["location"]: feature["properties"] for feature in geojson["features"]}
    shared_countries = {"Spain", "Pakistan"}
    sturm = []
    for sensor, folder, time_column in (("S1", "Sentinel1", "sentinel_date"), ("S2", "Sentinel2", "sentinel_timestamp")):
        with (STURM_ROOT / "metadata" / f"{folder}_metadata.csv").open(newline="", encoding="utf-8") as handle:
            metadata = list(csv.DictReader(handle))
        for row in metadata:
            if row["country"] not in shared_countries:
                continue
            path = STURM_ROOT / f"external_overlap_{sensor.lower()}_masks" / row["tile_id"]
            if not path.exists():
                raise RuntimeError(f"Missing STURM source footprint: {path.name}")
            with rasterio.open(path) as source:
                bbox = transform_bounds(source.crs, "EPSG:4326", *source.bounds, densify_pts=21)
            sturm.append((sensor, row["country"], row["ems_code"], row["tile_id"], datetime.fromisoformat(row[time_column]), bbox))
    rows = []
    labels = sorted((SEN_ROOT / "data" / "flood_events" / "HandLabeled" / "LabelHand").glob("*.tif"))
    for path in labels:
        scene = path.name.removesuffix("_LabelHand.tif")
        event = scene.split("_", 1)[0]
        country = EVENT_COUNTRY.get(event, event)
        event_date = datetime.strptime(events[country]["s1_date"], "%Y/%m/%d")
        with rasterio.open(path) as source:
            bbox = transform_bounds(source.crs, "EPSG:4326", *source.bounds, densify_pts=21)
        same_country = [record for record in sturm if record[1] == country]
        spatial = [record for record in same_country if intersects(record[5], bbox)]
        same_s1 = [record for record in same_country if record[0] == "S1"]
        same_s2 = [record for record in same_country if record[0] == "S2"]
        overlap_s1 = [record for record in spatial if record[0] == "S1"]
        overlap_s2 = [record for record in spatial if record[0] == "S2"]
        rows.append({
            "sen1_scene": scene,
            "sen1_event": event,
            "sen1_country": country,
            "sen1_event_date": event_date.date().isoformat(),
            "sturm_s1_same_country_tiles": len(same_s1),
            "sturm_s2_same_country_tiles": len(same_s2),
            "sturm_s1_spatial_bbox_overlap_tiles": len(overlap_s1),
            "sturm_s2_spatial_bbox_overlap_tiles": len(overlap_s2),
            "sturm_any_spatial_bbox_overlap_tiles": len(spatial),
            "sturm_spatial_bbox_overlap_activations": ";".join(sorted({record[2] for record in spatial})),
            "min_day_gap_to_same_country_sturm_s1": min((abs((record[4] - event_date).total_seconds()) / 86400 for record in same_s1), default=None),
            "min_day_gap_to_same_country_sturm_s2": min((abs((record[4] - event_date).total_seconds()) / 86400 for record in same_s2), default=None),
            "min_day_gap_to_spatial_overlap_sturm_any": min((abs((record[4] - event_date).total_seconds()) / 86400 for record in spatial), default=None),
        })
    if len(rows) != 90:
        raise RuntimeError(f"Expected 90 test scenes, found {len(rows)}")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Audited {len(rows)} Sen1 test scenes against {len(sturm)} STURM S1/S2 masks")
    print("Scenes with spatial bounding-box overlap:", sum(row["sturm_any_spatial_bbox_overlap_tiles"] > 0 for row in rows))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
