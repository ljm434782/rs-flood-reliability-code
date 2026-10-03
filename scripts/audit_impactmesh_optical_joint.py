"""Audit joint optical visibility for every S2 time in the fixed train pilot."""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
FRAME = MANIFESTS / "impactmesh_temporal_audit_frame.csv"
SAMPLES = ROOT / "data" / "raw" / "impactmesh" / "samples"
OUTPUT = MANIFESTS / "impactmesh_optical_joint_audit.csv"
META = MANIFESTS / "impactmesh_optical_joint_audit_meta.json"
PIXELS = 256 * 256


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def audit_count(table: dict[str, int], allowed: set[int]) -> int:
    return sum(int(value) for key, value in table.items() if float(key) in allowed)


def main() -> None:
    with FRAME.open(newline="", encoding="utf-8") as handle:
        frame = list(csv.DictReader(handle))
    if len(frame) != 24 or len({row["activation"] for row in frame}) != 24:
        raise RuntimeError("Expected 24 distinct frozen train activations")
    rows = []
    fields = [
        "rank", "activation", "sample_id", "time_index", "s1_event_utc", "s2_utc",
        "s2_minus_s1_days", "s2_at_or_before_s1", "data_mask_value", "pixels",
        "scl_4_5_6_pixels", "cloud_mask_0_pixels", "nan_mask_0_pixels",
        "all_12_bands_finite_nonfill_pixels", "joint_visible_pixels", "joint_visible_fraction",
    ]
    for selected in frame:
        sample_id = selected["sample_id"]
        audit = json.loads((MANIFESTS / f"impactmesh_sample_audit_{sample_id}.json").read_text(encoding="utf-8"))
        if audit["split"] != "train" or audit["sample_id"] != sample_id:
            raise RuntimeError(f"Sample disagrees with train frame: {sample_id}")
        event_s1 = utc(audit["S1RTC"]["timestamps_utc"][2])
        path = SAMPLES / sample_id / f"{sample_id}_S2L2A.zarr.zip"
        group = zarr.open_consolidated(path, mode="r")
        times = group.attrs["datetime"].split(";")
        if times != audit["S2L2A"]["timestamps_utc"] or list(group["bands"].shape) != [4, 12, 256, 256]:
            raise RuntimeError(f"S2 timestamps or shape disagree with saved audit: {sample_id}")
        for index, value in enumerate(times):
            scl = group["SCL"][index]
            cloud = group["cloud_mask"][index]
            nan = group["nan_mask"][index]
            bands = group["bands"][index]
            data = int(group["data_mask"][index])
            if scl.shape != cloud.shape or scl.shape != nan.shape or scl.shape != (256, 256) or data not in (0, 1):
                raise RuntimeError(f"Unexpected S2 mask shape or data flag: {sample_id}, time {index}")
            visible_scl = np.isin(scl, (4, 5, 6))
            clear_cloud = cloud == 0
            valid_nan = nan == 0
            valid_bands = np.isfinite(bands).all(axis=0) & (bands != -9999).all(axis=0)
            joint = visible_scl & clear_cloud & valid_nan & valid_bands & (data == 1)
            prior = audit["S2L2A"]
            if (count(visible_scl) != audit_count(prior["SCL_value_counts_by_time"][index], {4, 5, 6}) or
                    count(clear_cloud) != audit_count(prior["cloud_mask_value_counts_by_time"][index], {0}) or
                    count(valid_nan) != audit_count(prior["nan_mask_value_counts_by_time"][index], {0}) or
                    data != prior["data_mask_values"][index]):
                raise RuntimeError(f"Raw S2 masks disagree with saved marginal audit: {sample_id}, time {index}")
            timestamp = utc(value)
            rows.append({
                "rank": selected["rank"],
                "activation": selected["activation"],
                "sample_id": sample_id,
                "time_index": index,
                "s1_event_utc": event_s1.isoformat().replace("+00:00", "Z"),
                "s2_utc": value,
                "s2_minus_s1_days": (timestamp - event_s1).total_seconds() / 86400,
                "s2_at_or_before_s1": int(timestamp <= event_s1),
                "data_mask_value": data,
                "pixels": PIXELS,
                "scl_4_5_6_pixels": count(visible_scl),
                "cloud_mask_0_pixels": count(clear_cloud),
                "nan_mask_0_pixels": count(valid_nan),
                "all_12_bands_finite_nonfill_pixels": count(valid_bands),
                "joint_visible_pixels": count(joint),
                "joint_visible_fraction": count(joint) / PIXELS,
            })
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    META.write_text(json.dumps({
        "frozen_frame_sha256": hashlib.sha256(FRAME.read_bytes()).hexdigest(),
        "source": "24 fixed train samples; four S2 times per sample; local verified member ZIPs",
        "rows": len(rows),
        "pixels_per_row": PIXELS,
        "joint_visible_definition": "SCL in {4,5,6} AND cloud_mask==0 AND nan_mask==0 AND all 12 bands finite and !=-9999 AND data_mask==1",
        "use_limit": "descriptive training-source audit only; no age/coverage eligibility threshold, no process-pass rate or main-data selection",
        "scl_reference": "https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html",
    }, indent=2), encoding="utf-8")
    print(f"Audited {len(rows)} S2 observations from {len(frame)} fixed train samples")
    print(OUTPUT)


if __name__ == "__main__":
    main()
