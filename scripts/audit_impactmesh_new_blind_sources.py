"""Input-only inventory of never evaluated official val/test activation candidates.

This script reads split sample IDs, mask georeferencing, and a cached CEMS API
snapshot. It never opens label pixels or model output.
"""

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.train_impactmesh_formal import ROOT, sha256

MAN = ROOT / "data/manifests"
API = ROOT / "data/raw/cems/activations_api_20260927.json"
OUT = MAN / "impactmesh_new_blind_source_inventory.csv"
META = OUT.with_suffix(".json")


def sample_counts(split):
    path = MAN / f"impactmesh_flood_{split}.txt"
    return Counter(line.split("_")[0] for line in path.read_text(encoding="utf-8").splitlines() if line)


def parse_point(wkt):
    match = re.fullmatch(r"POINT \(([-.\d]+) ([-.\d]+)\)", wkt or "")
    return (float(match[1]), float(match[2])) if match else None


def haversine_km(a, b):
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    argument = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 12742 * math.asin(min(1, math.sqrt(argument)))


def main():
    counts = {role: sample_counts(role) for role in ("train", "val", "test", "test_holdout")}
    candidates = sorted((set(counts["val"]) | set(counts["test"])) -
                        set(counts["train"]) - set(counts["test_holdout"]))
    api = json.loads(API.read_text(encoding="utf-8"))
    all_events = {row["code"]: row for row in api["results"]}
    if len(candidates) != 12 or any(code not in all_events for code in candidates):
        raise RuntimeError("Official candidate inventory or CEMS metadata changed")
    seen = set(counts["train"]) | set(counts["test_holdout"])
    rows = []
    for code in candidates:
        event = all_events[code]
        instant = datetime.fromisoformat(event["activationTime"])
        point = parse_point(event["centroid"])
        near = []
        for other in sorted(seen):
            if other not in all_events:
                continue
            ref = all_events[other]
            if not ref.get("activationTime") or point is None:
                continue
            days = abs((instant - datetime.fromisoformat(ref["activationTime"])).total_seconds()) / 86400
            ref_point = parse_point(ref.get("centroid"))
            if ref_point is None:
                continue
            distance = haversine_km(point, ref_point)
            if days <= 30 and distance <= 500:
                near.append(f"{other}:{days:.1f}d:{distance:.0f}km")
        rows.append({"activation": code, "name": event["name"],
                     "event_type": event["category"]["name"],
                     "countries": ";".join(c["short_name"] for c in event["countries"]),
                     "activation_time": event["activationTime"],
                     "centroid_wkt": event["centroid"],
                     "val_input_sample_count": counts["val"][code],
                     "test_input_sample_count": counts["test"][code],
                     "near_seen_activation_screen": ";".join(near),
                     "source_url": f"https://mapping.emergency.copernicus.eu/activations/{code}/",
                     "status": "INPUT_ONLY_SCREEN_NOT_ELIGIBILITY_DECISION"})
    if OUT.exists() or META.exists():
        raise RuntimeError("Refusing to overwrite input-only source inventory")
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    META.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
        "source_api": "https://mapping.emergency.copernicus.eu/activations/api/activations/?limit=2000",
        "api_snapshot_sha256": sha256(API),
        "split_sha256": {role: sha256(MAN / f"impactmesh_flood_{role}.txt") for role in counts},
        "output_sha256": sha256(OUT), "candidate_count": len(rows),
        "read_label_pixels": False, "read_model_outputs": False,
        "limitation": "activation centroids and 30d/500km screens do not establish process independence"},
        indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row["activation"], row["activation_time"], row["countries"],
              row["val_input_sample_count"], row["test_input_sample_count"],
              row["near_seen_activation_screen"])


if __name__ == "__main__":
    main()
