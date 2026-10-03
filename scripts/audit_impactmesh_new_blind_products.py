"""Read CEMS product-source times and AOI footprints for blind candidates.

No ImpactMesh label pixels or model predictions are opened. A close reference
image time does not prove which CEMS product generated a tile's mask.
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from shapely import wkt

from scripts.audit_impactmesh_new_blind_spatial import polygon, rows
from scripts.train_impactmesh_formal import ROOT, sha256

MAN = ROOT / "data/manifests"
RAW = ROOT / "data/raw/cems"
OUT = MAN / "impactmesh_new_blind_cems_product_time_screen.csv"
META = OUT.with_suffix(".json")
IDS = ("EMSR788", "EMSR843")
API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api/public-activations/?code={}"


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def download_detail(code):
    path = RAW / f"{code}_public_detail_20260927.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        response = requests.get(API.format(code), timeout=40)
        response.raise_for_status()
        data = response.json()
        if data.get("count") != 1 or data["results"][0]["code"] != code:
            raise RuntimeError(f"Unexpected CEMS public detail response: {code}")
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if data.get("count") != 1 or data["results"][0]["code"] != code:
        raise RuntimeError(f"Cached CEMS public detail mismatch: {code}")
    return data["results"][0], path


def main():
    if OUT.exists() or META.exists():
        raise RuntimeError("Refusing to overwrite product-time screen")
    details = {}
    paths = {}
    for code in IDS:
        details[code], paths[code] = download_detail(code)
    s1 = {row["sample_id"]: row for row in csv.DictReader(
        (MAN / "impactmesh_new_blind_s1_times.csv").open(newline="", encoding="utf-8"))}
    candidates = [row for row in rows("val") if row["activation"] in IDS]
    transformed = {}
    records = []
    for row in candidates:
        code = row["activation"]
        shape = polygon(row, transformed)
        reference = timestamp(s1[row["sample_id"]]["event_s1_utc"])
        matching_aois = []
        del_images = []
        all_images = []
        for aoi in details[code]["aois"]:
            footprint = wkt.loads(aoi["extent"])
            if shape.intersection(footprint).area <= 0:
                continue
            matching_aois.append(str(aoi["number"]))
            for product in aoi["products"]:
                for image in product.get("images", []):
                    if not image.get("new") or not image.get("acquisitionTime"):
                        continue
                    delta_hours = (timestamp(image["acquisitionTime"]) - reference).total_seconds() / 3600
                    all_images.append(delta_hours)
                    if product["type"] == "DEL":
                        del_images.append(delta_hours)
        closest = min(del_images, key=lambda value: abs(value)) if del_images else None
        records.append({"sample_id": row["sample_id"], "activation": code,
                        "event_s1_utc": s1[row["sample_id"]]["event_s1_utc"],
                        "overlapping_cems_aoi_numbers": ";".join(sorted(set(matching_aois))),
                        "matching_del_source_image_count": len(del_images),
                        "closest_del_source_delta_hours": "" if closest is None else closest,
                        "del_source_within_1h_of_s1": int(any(abs(value) <= 1 for value in del_images)),
                        "matching_any_product_source_image_count": len(all_images)})
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 144:
        raise RuntimeError("CEMS product-time candidate tile count changed")
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    by_event = defaultdict(list)
    for row in records:
        by_event[row["activation"]].append(row)
    summary = {code: {"tiles": len(items),
        "tiles_with_aoi_overlap": sum(bool(row["overlapping_cems_aoi_numbers"]) for row in items),
        "tiles_with_del_source_image": sum(row["matching_del_source_image_count"] > 0 for row in items),
        "tiles_with_del_source_within_1h": sum(row["del_source_within_1h_of_s1"] for row in items),
        "related_events": details[code].get("relatedevents", []),
        "cems_aoi_count": len(details[code]["aois"]),
        "cems_product_type_counts": dict(Counter(p["type"] for a in details[code]["aois"]
                                                for p in a["products"]))}
        for code, items in sorted(by_event.items())}
    META.write_text(json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(),
        "api_source": API, "raw_detail_sha256": {code: sha256(path) for code, path in paths.items()},
        "source_s1_times_sha256": sha256(MAN / "impactmesh_new_blind_s1_times.csv"),
        "source_grid_sha256": sha256(MAN / "impactmesh_val_mask_grid_index.csv"),
        "output_sha256": sha256(OUT), "event_summary": summary,
        "label_pixels_read": False, "model_outputs_read": False,
        "limitation": "closest CEMS product source image is not a verified per-tile mask product mapping"},
        indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
