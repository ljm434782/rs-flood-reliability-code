"""Summarize age/visibility tradeoffs in the frozen train-only optical pilot."""

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
FRAME = MANIFESTS / "impactmesh_temporal_audit_frame.csv"
INPUT = MANIFESTS / "impactmesh_optical_joint_audit.csv"
META = MANIFESTS / "impactmesh_optical_joint_audit_meta.json"
OUTPUT = MANIFESTS / "impactmesh_optical_window_sensitivity.json"
WINDOW_DAYS = (7, 14, 30, 45)


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def selected(row: dict[str, str]) -> dict[str, float | int | str]:
    return {
        "time_index": int(row["time_index"]),
        "s2_utc": row["s2_utc"],
        "age_days": -float(row["s2_minus_s1_days"]),
        "joint_visible_fraction": float(row["joint_visible_fraction"]),
    }


def main() -> None:
    frame_bytes = FRAME.read_bytes()
    frame_sha = hashlib.sha256(frame_bytes).hexdigest()
    meta = json.loads(META.read_text(encoding="utf-8"))
    if frame_sha != meta["frozen_frame_sha256"]:
        raise RuntimeError("Frozen frame hash differs from optical audit metadata")
    with FRAME.open(newline="", encoding="utf-8") as handle:
        frame = list(csv.DictReader(handle))
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(frame) != 24 or len(rows) != 96 or int(meta["rows"]) != len(rows):
        raise RuntimeError("Expected 24 frozen train samples and 96 optical rows")
    by_id = defaultdict(list)
    for row in rows:
        event_time = timestamp(row["s1_event_utc"])
        image_time = timestamp(row["s2_utc"])
        actual_delta = (image_time - event_time).total_seconds() / 86400
        delta = float(row["s2_minus_s1_days"])
        frac = float(row["joint_visible_fraction"])
        if (abs(actual_delta - delta) > 1e-8 or
                int(row["s2_at_or_before_s1"]) != int(image_time <= event_time) or
                int(row["pixels"]) != 65536 or
                int(row["joint_visible_pixels"]) / 65536 != frac or
                not 0 <= frac <= 1):
            raise RuntimeError(f"Inconsistent optical audit row: {row['sample_id']}")
        by_id[row["sample_id"]].append(row)
    if {row["sample_id"] for row in frame} != set(by_id):
        raise RuntimeError("Optical rows differ from frozen frame")
    for source in frame:
        sample_rows = by_id[source["sample_id"]]
        if (len(sample_rows) != 4 or
                {int(row["time_index"]) for row in sample_rows} != set(range(4)) or
                any(row["rank"] != source["rank"] or row["activation"] != source["activation"]
                    for row in sample_rows)):
            raise RuntimeError(f"Four time rows disagree with frozen frame: {source['sample_id']}")

    result = {
        "scope": "descriptive sensitivity on 24 preselected train chips; not an eligibility rule or process pass rate",
        "frozen_frame_sha256": frame_sha,
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "selection": "S2 acquired at or before event S1; age <= window days; freshest=max S2 time; clearest=max joint visible fraction, ties to fresher time",
        "windows": [],
    }
    for days in WINDOW_DAYS:
        samples = []
        for source in frame:
            candidates = [row for row in by_id[source["sample_id"]]
                          if 0 <= -float(row["s2_minus_s1_days"]) <= days]
            item = {"rank": int(source["rank"]), "activation": source["activation"],
                    "sample_id": source["sample_id"], "has_prior_s2_in_window": bool(candidates)}
            if candidates:
                freshest = max(candidates, key=lambda row: timestamp(row["s2_utc"]))
                clearest = max(candidates, key=lambda row: (
                    float(row["joint_visible_fraction"]), timestamp(row["s2_utc"])))
                item["freshest"] = selected(freshest)
                item["clearest"] = selected(clearest)
            samples.append(item)
        available = [item for item in samples if item["has_prior_s2_in_window"]]
        result["windows"].append({
            "window_days": days,
            "chips_with_prior_s2": len(available),
            "median_freshest_visible_fraction": median(item["freshest"]["joint_visible_fraction"] for item in available) if available else None,
            "median_clearest_visible_fraction": median(item["clearest"]["joint_visible_fraction"] for item in available) if available else None,
            "median_freshest_age_days": median(item["freshest"]["age_days"] for item in available) if available else None,
            "median_clearest_age_days": median(item["clearest"]["age_days"] for item in available) if available else None,
            "clearest_differs_from_freshest": sum(item["freshest"]["time_index"] != item["clearest"]["time_index"] for item in available),
            "samples": samples,
        })
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in result["windows"]:
        print(item["window_days"], item["chips_with_prior_s2"],
              item["median_freshest_visible_fraction"], item["median_clearest_visible_fraction"],
              item["clearest_differs_from_freshest"])
    print(OUTPUT)


if __name__ == "__main__":
    main()
