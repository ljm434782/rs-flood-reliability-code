"""Verify the fixed train pilot against local files and saved provenance."""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
FRAME = MANIFESTS / "impactmesh_temporal_audit_frame.csv"
PROGRESS = MANIFESTS / "impactmesh_temporal_pilot_progress.csv"
MANIFEST = MANIFESTS / "impactmesh_sample_manifest.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    frame = rows(FRAME)
    progress = rows(PROGRESS)
    manifest = rows(MANIFEST)
    frame_meta = json.loads((MANIFESTS / "impactmesh_temporal_audit_frame_meta.json").read_text(encoding="utf-8"))
    progress_meta = json.loads((MANIFESTS / "impactmesh_temporal_pilot_progress_meta.json").read_text(encoding="utf-8"))
    frame_hash = hashlib.sha256(FRAME.read_bytes()).hexdigest()
    require(frame_hash == progress_meta["frozen_frame_sha256"], "Frozen frame hash changed")
    require(len(frame) == len(progress) == 24, "Pilot must contain all 24 fixed rows")
    require(frame_meta["selected_activations"] == 24, "Frame metadata disagrees")
    require({row["rank"] for row in frame} == {str(rank) for rank in range(1, 25)}, "Frame ranks are incomplete")
    require(len({row["activation"] for row in frame}) == 24, "Frame activations are not distinct")
    require(progress_meta["audited_rows"] == 24 and progress_meta["pending_rows"] == 0, "Progress metadata is incomplete")
    event_after = 0
    prior_ages = []
    for selected, reported in zip(frame, progress):
        rank, sample_id = int(selected["rank"]), selected["sample_id"]
        require(all(reported[key] == selected[key] for key in ("rank", "activation", "sample_id")), f"Progress frame mismatch: rank {rank}")
        require(reported["state"] == "audited", f"Unaudited sample: rank {rank}")
        audit_path = MANIFESTS / f"impactmesh_sample_audit_{sample_id}.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        require(audit["sample_id"] == sample_id and audit["split"] == "train", f"Audit identity mismatch: rank {rank}")
        require(f"frame rank {rank}" in audit["selection"], f"Audit selection mismatch: rank {rank}")
        require(audit["s1_s2_coordinates_match"] and audit["label"]["sensor_grid_matches_label_centers"], f"Grid mismatch: rank {rank}")
        event_s1 = datetime.fromisoformat(audit["S1RTC"]["timestamps_utc"][2].replace("Z", "+00:00"))
        s2_times = [datetime.fromisoformat(value.replace("Z", "+00:00")) for value in audit["S2L2A"]["timestamps_utc"]]
        prior = [index for index, timestamp in enumerate(s2_times) if timestamp <= event_s1]
        require(bool(prior), f"No prior S2 for rank {rank}")
        closest = max(prior, key=lambda index: s2_times[index])
        event_delta = (s2_times[2] - event_s1).total_seconds() / 86400
        prior_delta = (s2_times[closest] - event_s1).total_seconds() / 86400
        event_after += event_delta > 0
        prior_ages.append(-prior_delta)
        require(reported["event_s1_utc"] == event_s1.isoformat().replace("+00:00", "Z"), f"S1 timestamp mismatch: rank {rank}")
        require(abs(float(reported["event_s2_minus_s1_days"]) - event_delta) < 1e-9, f"Event time delta mismatch: rank {rank}")
        require(int(reported["closest_prior_s2_index"]) == closest, f"Prior S2 index mismatch: rank {rank}")
        require(abs(float(reported["closest_prior_s2_minus_s1_days"]) - prior_delta) < 1e-9, f"Prior S2 time delta mismatch: rank {rank}")
        for column, source in (("cloudmask", "cloud_mask_value_counts_by_time"),
                               ("scl", "SCL_value_counts_by_time"),
                               ("nanmask", "nan_mask_value_counts_by_time")):
            require(json.loads(reported[f"closest_prior_{column}_counts_json"]) == audit["S2L2A"][source][closest], f"{column} count mismatch: rank {rank}")
        require(reported["closest_prior_data_mask_value"] == str(audit["S2L2A"]["data_mask_values"][closest]), f"Data mask mismatch: rank {rank}")
        for modality in ("MASK", "S1RTC", "S2L2A"):
            matching = [row for row in manifest if row["sample_id"] == sample_id and row["split"] == "train" and row["modality"] == modality]
            require(len(matching) == 1, f"Expected one {modality} manifest row: rank {rank}")
            item = matching[0]
            require(item["tar_path"] == f"train/{modality}.tar", f"Wrong TAR: rank {rank} {modality}")
            require(int(item["tar_ordinal"]) == int(selected["train_mask_tar_ordinal"]), f"TAR ordinal mismatch: rank {rank} {modality}")
            local = (ROOT / item["local_path"]).resolve()
            require(local.is_relative_to(ROOT), f"Local path escaped workspace: rank {rank} {modality}")
            require(local.is_file(), f"Missing local member: rank {rank} {modality}")
            require(local.stat().st_size == int(item["size_bytes"]), f"Local size mismatch: rank {rank} {modality}")
            require(hashlib.sha256(local.read_bytes()).hexdigest() == item["sha256"], f"Local SHA256 mismatch: rank {rank} {modality}")
            if modality != "MASK":
                location = json.loads((MANIFESTS / f"impactmesh_tar_location_rank{rank}_{modality}.json").read_text(encoding="utf-8"))
                require(location["sample_id"] == sample_id and location["frame_rank"] == rank, f"Location identity mismatch: rank {rank} {modality}")
                require(location["tar_path"] == item["tar_path"], f"Location TAR mismatch: rank {rank} {modality}")
                require(location["member_offset_bytes"] == int(item["member_offset_bytes"]) and location["member_size_bytes"] == int(item["size_bytes"]), f"Location member mismatch: rank {rank} {modality}")
    descriptive = progress_meta["audited_sample_temporal_descriptives"]
    require(descriptive["event_s2_after_s1"] == event_after, "Event S2 after-S1 count mismatch")
    require(descriptive["event_s2_at_or_before_s1"] == 24 - event_after, "Event S2 prior count mismatch")
    require(descriptive["samples_with_prior_s2"] == 24, "Prior S2 count mismatch")
    for label, actual in (("min", min(prior_ages)), ("median", median(prior_ages)), ("max", max(prior_ages))):
        require(abs(descriptive[f"closest_prior_s2_age_days_{label}"] - actual) < 1e-9, f"Prior S2 age {label} mismatch")
    print("Verified 24 fixed train activations, 24 audit records, and 72 local TAR members by size/SHA256")
    print("Verified frame/progress timing and mask counts, grids, recorded ordinal consistency, and S1/S2 location records")
    print("Scope: local pilot provenance only; full S1/S2 TAR hashes and CEMS label-time chains remain unverified")


if __name__ == "__main__":
    main()
