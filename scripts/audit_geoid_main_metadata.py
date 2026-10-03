"""Input-only GEOID main-tree event overlap screen; never opens label rasters."""

import csv
import json
import math
import re
from datetime import datetime, timezone

import pandas as pd

from scripts.train_impactmesh_formal import ROOT, SPLIT, sha256

CATALOG = ROOT / "data/raw/geoid-flood/tile_catalog.parquet"
CEMS = ROOT / "data/raw/cems/activations_api_20260927_limit1000.json"
VIEWED_PANEL = ROOT / "data/manifests/impactmesh_new_blind_locked_tiles.json"
OUT = ROOT / "data/manifests/geoid_main_input_only_event_screen.csv"
META = OUT.with_suffix(".json")
POINT = re.compile(r"POINT\s*\((-?[\d.]+)\s+(-?[\d.]+)\)")


def coords(value):
    match = POINT.fullmatch(value or "")
    return (float(match.group(1)), float(match.group(2))) if match else None


def haversine_km(a, b):
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    h = (math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2)
         * math.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0088 * 2 * math.asin(min(1, math.sqrt(h)))


def main():
    if OUT.exists() or META.exists():
        raise RuntimeError("Refusing to overwrite GEOID main input audit")
    table = pd.read_parquet(CATALOG)
    table["publisher_eligible"] = table.is_valid & (table.invalid_pixel_frac <= .95)
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        prior = {row["activation"]: row["role"] for row in csv.DictReader(handle)}
    viewed = set(json.loads(VIEWED_PANEL.read_text(encoding="utf-8"))["locked_events"])
    source = json.loads(CEMS.read_text(encoding="utf-8"))
    events = {row["code"]: row for row in source["results"]}
    if set(table.event_id) - set(events):
        raise RuntimeError("CEMS metadata snapshot missing GEOID events")
    seen = set(prior) | viewed
    seen_meta = [(code, events[code]) for code in seen if code in events]
    rows = []
    for event_id, group in table.groupby("event_id"):
        source_row = events[event_id]
        candidate_time = pd.to_datetime(source_row["activationTime"], utc=True)
        candidate_point = coords(source_row.get("centroid"))
        near = []
        for seen_id, seen_row in seen_meta:
            if seen_id == event_id or candidate_point is None:
                continue
            point = coords(seen_row.get("centroid"))
            if point is None:
                continue
            days = abs((candidate_time - pd.to_datetime(seen_row["activationTime"], utc=True)).total_seconds()) / 86400
            if days <= 30:
                km = haversine_km(candidate_point, point)
                if km <= 500:
                    near.append((days, km, seen_id))
        near.sort()
        rows.append({
            "activation": event_id,
            "publisher_eligible_tiles": int(group.publisher_eligible.sum()),
            "all_catalog_tiles": len(group),
            "publisher_splits": ";".join(sorted(set(group.split))),
            "impactmesh_role": prior.get(event_id, "ABSENT"),
            "previously_viewed_four_event_panel": int(event_id in viewed),
            "cems_name": source_row.get("name", ""),
            "cems_category": (source_row.get("category") or {}).get("name", ""),
            "cems_activation_time": source_row.get("activationTime", ""),
            "cems_countries": ";".join(sorted(c.get("short_name", "") for c in source_row.get("countries", []))),
            "cems_centroid_wkt": source_row.get("centroid", ""),
            "near_seen_30d_500km_count": len(near),
            "near_seen_30d_500km_examples": ";".join(
                f"{code}:{days:.1f}d:{km:.0f}km" for days, km, code in near[:5]),
            "input_only_candidate_ge5": int(
                event_id not in seen and int(group.publisher_eligible.sum()) >= 5),
        })
    rows.sort(key=lambda row: row["activation"])
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidate = [row for row in rows if row["input_only_candidate_ge5"]]
    META.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "GEOID main tile catalog and CEMS metadata only; no imagery, label pixels or model scores",
        "publisher_revision": "868407460bf3db492f50730a57585916baa71dc6",
        "publisher_dataset_license": "CC BY 4.0 with Copernicus notices",
        "catalog_sha256": sha256(CATALOG), "cems_api_snapshot_sha256": sha256(CEMS),
        "impactmesh_split_sha256": sha256(SPLIT), "viewed_panel_sha256": sha256(VIEWED_PANEL),
        "output_sha256": sha256(OUT),
        "main_catalog_activations": len(rows),
        "main_publisher_eligible_activations": sum(row["publisher_eligible_tiles"] > 0 for row in rows),
        "exact_id_absent_from_impactmesh_split": sum(row["impactmesh_role"] == "ABSENT" for row in rows),
        "unseen_exact_id_with_at_least_five_publisher_eligible_tiles": len(candidate),
        "candidate_near_seen_screen_count": sum(row["near_seen_30d_500km_count"] > 0 for row in candidate),
        "candidate_tiles_before_source_time_unit_audit": sum(row["publisher_eligible_tiles"] for row in candidate),
        "test_labels_or_metrics_read": False,
        "warning": "Centroid/time screen is not process identity proof or final evaluation eligibility",
    }, indent=2) + "\n", encoding="utf-8")
    print(META.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
