"""Report progress and raw timing/cloud counts for all 24 fixed train pilot rows."""

import csv
import hashlib
import json
from statistics import median
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
FRAME = MANIFESTS / "impactmesh_temporal_audit_frame.csv"
OUTPUT = MANIFESTS / "impactmesh_temporal_pilot_progress.csv"
META = MANIFESTS / "impactmesh_temporal_pilot_progress_meta.json"


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    with FRAME.open(newline="", encoding="utf-8") as handle:
        frame = list(csv.DictReader(handle))
    if len(frame) != 24 or len({row["activation"] for row in frame}) != 24:
        raise RuntimeError("Frozen pilot frame is not 24 distinct activations")
    fields = [
        "rank", "activation", "sample_id", "state", "event_s1_utc",
        "event_s2_minus_s1_days", "closest_prior_s2_index",
        "closest_prior_s2_minus_s1_days", "closest_prior_cloudmask_counts_json",
        "closest_prior_scl_counts_json", "closest_prior_nanmask_counts_json",
        "closest_prior_data_mask_value",
    ]
    rows = []
    for selected in frame:
        row = {field: "" for field in fields}
        row.update({key: selected[key] for key in ("rank", "activation", "sample_id")})
        path = MANIFESTS / f"impactmesh_sample_audit_{selected['sample_id']}.json"
        if not path.exists():
            row["state"] = "pending"
            rows.append(row)
            continue
        audit = json.loads(path.read_text(encoding="utf-8"))
        if (audit["sample_id"] != selected["sample_id"] or audit["split"] != "train" or
                f"frame rank {selected['rank']}" not in audit["selection"] or
                not audit["s1_s2_coordinates_match"] or
                not audit["label"]["sensor_grid_matches_label_centers"]):
            raise RuntimeError(f"Audited sample disagrees with frozen frame: {selected['sample_id']}")
        event_s1 = parse_utc(audit["S1RTC"]["timestamps_utc"][2])
        s2_times = [parse_utc(value) for value in audit["S2L2A"]["timestamps_utc"]]
        prior = [index for index, timestamp in enumerate(s2_times) if timestamp <= event_s1]
        row.update({
            "state": "audited",
            "event_s1_utc": event_s1.isoformat().replace("+00:00", "Z"),
            "event_s2_minus_s1_days": (s2_times[2] - event_s1).total_seconds() / 86400,
        })
        if prior:
            closest = max(prior, key=lambda index: s2_times[index])
            cloud = audit["S2L2A"]["cloud_mask_value_counts_by_time"][closest]
            scl = audit["S2L2A"]["SCL_value_counts_by_time"][closest]
            nan = audit["S2L2A"]["nan_mask_value_counts_by_time"][closest]
            if any(sum(int(value) for value in table.values()) != 256 * 256 for table in (cloud, scl, nan)):
                raise RuntimeError(f"Incomplete pixel counts for {selected['sample_id']}")
            row.update({
                "closest_prior_s2_index": closest,
                "closest_prior_s2_minus_s1_days": (s2_times[closest] - event_s1).total_seconds() / 86400,
                "closest_prior_cloudmask_counts_json": json.dumps(cloud, sort_keys=True),
                "closest_prior_scl_counts_json": json.dumps(scl, sort_keys=True),
                "closest_prior_nanmask_counts_json": json.dumps(nan, sort_keys=True),
                "closest_prior_data_mask_value": audit["S2L2A"]["data_mask_values"][closest],
            })
        rows.append(row)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    audited = [row for row in rows if row["state"] == "audited"]
    prior_ages = [-float(row["closest_prior_s2_minus_s1_days"]) for row in audited if row["closest_prior_s2_index"] != ""]
    META.write_text(json.dumps({
        "frozen_frame_sha256": hashlib.sha256(FRAME.read_bytes()).hexdigest(),
        "fixed_frame_rows": len(rows),
        "audited_rows": len(audited),
        "pending_rows": sum(row["state"] == "pending" for row in rows),
        "time_anchor": "S1 index 2; closest prior S2 chosen by timestamp across all four S2 indices",
        "audited_sample_temporal_descriptives": {
            "event_s2_after_s1": sum(float(row["event_s2_minus_s1_days"]) > 0 for row in audited),
            "event_s2_at_or_before_s1": sum(float(row["event_s2_minus_s1_days"]) <= 0 for row in audited),
            "samples_with_prior_s2": len(prior_ages),
            "closest_prior_s2_age_days_min": min(prior_ages) if prior_ages else None,
            "closest_prior_s2_age_days_median": median(prior_ages) if prior_ages else None,
            "closest_prior_s2_age_days_max": max(prior_ages) if prior_ages else None,
        },
        "scope": "raw train-only pilot counts; no cloud eligibility cutoff, process-pass estimate, model result, or test data",
    }, indent=2), encoding="utf-8")
    print(f"Audited {sum(row['state'] == 'audited' for row in rows)}/24 fixed train samples; pending rows retained")
    print(OUTPUT)


if __name__ == "__main__":
    main()
