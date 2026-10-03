"""Apply frozen optical eligibility to all official-train inputs and split processes.

This runs only after the eligibility and split algorithms were committed. It
does not read label pixels or any official test/holdout imagery.
"""

import csv
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr
from rasterio.crs import CRS

from flood_reliability.splits import canonical_process_id, require_process_disjoint


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
ARCHIVES = ROOT / "data" / "raw" / "impactmesh" / "train"
SAMPLE_OUTPUT = MANIFESTS / "impactmesh_all_train_frozen_eligibility.csv.gz"
PROCESS_OUTPUT = MANIFESTS / "impactmesh_all_train_frozen_process_counts.csv"
SPLIT_OUTPUT = MANIFESTS / "impactmesh_process_split_frozen.csv"
META = MANIFESTS / "impactmesh_all_train_frozen_eligibility_meta.json"
MAX_AGE_DAYS = 30.0
MIN_VISIBLE_FRACTION = 0.5
MIN_PROCESS_SAMPLES = 5
SPLIT_SEED = "impactmesh-v1-final-process-split-20260927"
SUFFIX = {"S1RTC": "_S1RTC.zarr.zip", "S2L2A": "_S2L2A.zarr.zip"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def store_from_zip(blob: bytes, *, full: bool) -> zarr.storage.MemoryStore:
    store = zarr.storage.MemoryStore()
    prefixes = ("spatial_ref/", "x/", "y/", "data_mask/", "nan_mask/")
    if not full:
        prefixes += ("SCL/", "cloud_mask/")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in archive.namelist():
            if full or name in (".zmetadata", ".zattrs", ".zgroup") or name.startswith(prefixes):
                store[name] = archive.read(name)
    return store


def grid_fingerprint(group) -> str:
    digest = hashlib.sha256()
    crs = CRS.from_wkt(group["spatial_ref"].attrs["crs_wkt"])
    digest.update(crs.to_wkt().encode("utf-8"))
    digest.update(np.asarray(group["x"][:], dtype=np.float64).tobytes())
    digest.update(np.asarray(group["y"][:], dtype=np.float64).tobytes())
    return digest.hexdigest()


def member_id(member: tarfile.TarInfo, modality: str) -> str:
    suffix = SUFFIX[modality]
    if not member.isfile() or not member.name.startswith(f"{modality}/") or not member.name.endswith(suffix):
        raise RuntimeError(f"Unexpected {modality} member: {member.name}")
    return Path(member.name).name.removesuffix(suffix)


def main() -> None:
    freeze_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prereg_sha = sha256(ROOT / "docs" / "PREREGISTRATION.md")
    assets = {item["path"]: item for item in json.loads(
        (MANIFESTS / "impactmesh_assets.json").read_text(encoding="utf-8"))["assets"]}
    official = set((MANIFESTS / "impactmesh_flood_train.txt").read_text(encoding="utf-8").splitlines())
    holdout_ids = (MANIFESTS / "impactmesh_flood_test_holdout.txt").read_text(encoding="utf-8").splitlines()
    pilot = set()
    with (MANIFESTS / "impactmesh_temporal_audit_frame.csv").open(newline="", encoding="utf-8") as handle:
        pilot = {row["activation"] for row in csv.DictReader(handle)}
    if len(official) != 19448 or len(pilot) != 24:
        raise RuntimeError("Official train or fixed pilot count changed")

    s1 = {}
    s1_path = ARCHIVES / "S1RTC.tar"
    with tarfile.open(s1_path, "r:") as archive:
        for index, member in enumerate(archive, start=1):
            sample_id = member_id(member, "S1RTC")
            if sample_id not in official or sample_id in s1:
                raise RuntimeError(f"Unexpected/duplicate S1 sample: {sample_id}")
            group = zarr.open_consolidated(
                store_from_zip(archive.extractfile(member).read(), full=False), mode="r")
            times = group.attrs["datetime"].split(";")
            if len(times) != 4 or list(group["bands"].shape) != [4, 2, 256, 256]:
                raise RuntimeError(f"Unexpected S1 metadata: {sample_id}")
            s1[sample_id] = {
                "event_utc": times[2],
                "data_mask": int(group["data_mask"][2]),
                "nan_pixels": int(np.count_nonzero(group["nan_mask"][2])),
                "grid": grid_fingerprint(group),
            }
            if index % 1000 == 0:
                print(f"S1 full official-train inputs: {index}/{len(official)}", flush=True)
    if set(s1) != official:
        raise RuntimeError("S1 archive differs from official train split")

    sample_rows = []
    seen = set()
    s2_path = ARCHIVES / "S2L2A.tar"
    with tarfile.open(s2_path, "r:") as archive:
        for index, member in enumerate(archive, start=1):
            sample_id = member_id(member, "S2L2A")
            if sample_id not in official or sample_id in seen:
                raise RuntimeError(f"Unexpected/duplicate S2 sample: {sample_id}")
            seen.add(sample_id)
            group = zarr.open_consolidated(
                store_from_zip(archive.extractfile(member).read(), full=True), mode="r")
            times = group.attrs["datetime"].split(";")
            if len(times) != 4 or list(group["bands"].shape) != [4, 12, 256, 256]:
                raise RuntimeError(f"Unexpected S2 metadata: {sample_id}")
            source = s1[sample_id]
            grid_match = grid_fingerprint(group) == source["grid"]
            candidates = []
            for time_index, value in enumerate(times):
                age = (utc(source["event_utc"]) - utc(value)).total_seconds() / 86400
                if not 0 <= age <= MAX_AGE_DAYS:
                    continue
                scl = np.asarray(group["SCL"][time_index])
                cloud = np.asarray(group["cloud_mask"][time_index])
                nan = np.asarray(group["nan_mask"][time_index])
                bands = np.asarray(group["bands"][time_index])
                data = int(group["data_mask"][time_index])
                joint = (np.isin(scl, (4, 5, 6)) & (cloud == 0) & (nan == 0) &
                         np.all(np.isfinite(bands) & (bands != -9999), axis=0) & (data == 1))
                fraction = float(np.count_nonzero(joint) / 65536)
                candidates.append((fraction, -age, time_index, value, age))
            best = max(candidates) if candidates else None
            reasons = []
            if source["data_mask"] != 1:
                reasons.append("s1_data_mask")
            if source["nan_pixels"] != 0:
                reasons.append("s1_nan")
            if not grid_match:
                reasons.append("grid_mismatch")
            if best is None:
                reasons.append("no_pre_s1_s2_within_30d")
            elif best[0] < MIN_VISIBLE_FRACTION:
                reasons.append("visible_below_0.5")
            eligible = not reasons
            sample_rows.append({
                "sample_id": sample_id,
                "activation": sample_id.split("_", 1)[0],
                "paired_eligible": int(eligible),
                "selected_time_index": "" if best is None else best[2],
                "selected_s2_utc": "" if best is None else best[3],
                "selected_age_days": "" if best is None else best[4],
                "selected_joint_visible_fraction": "" if best is None else best[0],
                "failure_reasons": ";".join(reasons),
            })
            if index % 1000 == 0:
                print(f"S2 full official-train inputs: {index}/{len(official)}", flush=True)
    if seen != official:
        raise RuntimeError("S2 archive differs from official train split")

    sample_rows.sort(key=lambda row: row["sample_id"])
    sample_temp = SAMPLE_OUTPUT.with_suffix(".csv.gz.part")
    with gzip.open(sample_temp, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    sample_temp.replace(SAMPLE_OUTPUT)

    totals = Counter(row["activation"] for row in sample_rows)
    eligible_counts = Counter(row["activation"] for row in sample_rows if row["paired_eligible"])
    activations = sorted(totals, key=canonical_process_id)
    process_rows = [{
        "activation": activation,
        "official_train_samples": totals[activation],
        "paired_eligible_samples": eligible_counts[activation],
        "paired_process_eligible": int(eligible_counts[activation] >= MIN_PROCESS_SAMPLES),
        "fixed_pilot": int(activation in pilot),
    } for activation in activations]
    with PROCESS_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(process_rows[0]))
        writer.writeheader()
        writer.writerows(process_rows)

    eligible_processes = {row["activation"] for row in process_rows if row["paired_process_eligible"]}
    eligible_nonpilot = sorted(
        eligible_processes - pilot,
        key=lambda activation: hashlib.sha256(
            f"{SPLIT_SEED}|{canonical_process_id(activation)}".encode("ascii")).hexdigest())
    if len(eligible_nonpilot) < 60:
        raise RuntimeError("Fewer than 60 eligible non-pilot processes; frozen stop rule triggered")
    assigned = {}
    for activation in pilot & eligible_processes:
        assigned[activation] = "train"
    for index, activation in enumerate(eligible_nonpilot):
        assigned[activation] = ("internal_test" if index < 20 else
                                "calibration" if index < 40 else
                                "validation" if index < 60 else "train")
    split_rows = []
    for row in process_rows:
        activation = row["activation"]
        split_rows.append({
            **row,
            "role": assigned.get(activation, "excluded_paired"),
            "split_hash": hashlib.sha256(
                f"{SPLIT_SEED}|{canonical_process_id(activation)}".encode("ascii")).hexdigest(),
            "source": "official_train",
        })
    holdout_activations = sorted({sample_id.split("_", 1)[0] for sample_id in holdout_ids})
    for activation in holdout_activations:
        split_rows.append({
            "activation": activation,
            "official_train_samples": 0,
            "paired_eligible_samples": "",
            "paired_process_eligible": "",
            "fixed_pilot": 0,
            "role": "official_test_holdout",
            "split_hash": "",
            "source": "official_test_holdout",
        })
    groups = {}
    for role in ("train", "validation", "calibration", "internal_test", "official_test_holdout"):
        groups[role] = [row["activation"] for row in split_rows if row["role"] == role]
    require_process_disjoint(groups)
    with SPLIT_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)

    role_counts = Counter(row["role"] for row in split_rows)
    META.write_text(json.dumps({
        "freeze_commit_before_held_role_scan": freeze_commit,
        "preregistration_sha256": prereg_sha,
        "official_train_sample_count": len(official),
        "official_train_process_count": len(activations),
        "paired_eligible_sample_count": sum(row["paired_eligible"] for row in sample_rows),
        "paired_eligible_process_count": len(eligible_processes),
        "frozen_rule": {
            "max_pre_s1_age_days": MAX_AGE_DAYS,
            "minimum_joint_visible_fraction": MIN_VISIBLE_FRACTION,
            "minimum_samples_per_process": MIN_PROCESS_SAMPLES,
            "selection": "maximum joint visible fraction; ties to fresher S2",
        },
        "role_process_counts": dict(role_counts),
        "s1_tar_sha256": assets["train/S1RTC.tar"]["lfs"]["oid"],
        "s2_tar_sha256": assets["train/S2L2A.tar"]["lfs"]["oid"],
        "sample_output_sha256": sha256(SAMPLE_OUTPUT),
        "process_output_sha256": sha256(PROCESS_OUTPUT),
        "split_output_sha256": sha256(SPLIT_OUTPUT),
        "labels_read": False,
        "official_test_or_holdout_imagery_read": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(META.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
