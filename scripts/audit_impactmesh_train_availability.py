"""Inventory candidate training-role S1/S2 times and mask-based visibility.

This reads only the label-blind candidate training role within verified train
archives. Internal test, calibration and validation are untouched before
eligibility rules freeze. Visibility is an upper bound because S2 band values
are not checked here. It never selects or excludes samples.
"""

import csv
import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr
from rasterio.crs import CRS


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
ARCHIVES = ROOT / "data" / "raw" / "impactmesh" / "train"
OUTPUT = MANIFESTS / "impactmesh_candidate_train_s2_availability.csv.gz"
META = MANIFESTS / "impactmesh_candidate_train_s2_availability_meta.json"
SPLIT = MANIFESTS / "impactmesh_process_split_candidate.csv"
SUFFIX = {"S1RTC": "_S1RTC.zarr.zip", "S2L2A": "_S2L2A.zarr.zip"}
FIELDS = ["sample_id", "activation", "time_index", "s1_event_utc", "s2_utc",
          "s2_minus_s1_days", "s2_at_or_before_s1", "s1_event_data_mask",
          "s1_event_nan_pixels", "s1_s2_grid_match", "s2_data_mask",
          "scl_4_5_6_pixels", "cloud_mask_0_pixels", "nan_mask_0_pixels",
          "mask_joint_visible_upper_pixels", "mask_joint_visible_upper_fraction"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def verified_archive(modality: str, assets: dict) -> Path:
    path = ARCHIVES / f"{modality}.tar"
    asset = assets[f"train/{modality}.tar"]
    if path.stat().st_size != int(asset["size"]) or sha256(path) != asset["lfs"]["oid"]:
        raise RuntimeError(f"Full official archive has not passed SHA256: {path}")
    return path


def masked_store(blob: bytes, modality: str) -> zarr.storage.MemoryStore:
    store = zarr.storage.MemoryStore()
    prefixes = ("spatial_ref/", "x/", "y/", "data_mask/", "nan_mask/")
    if modality == "S2L2A":
        prefixes += ("SCL/", "cloud_mask/")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in archive.namelist():
            if name in (".zmetadata", ".zattrs", ".zgroup") or name.startswith(prefixes):
                store[name] = archive.read(name)
    return store


def member_id(member: tarfile.TarInfo, modality: str) -> str:
    suffix = SUFFIX[modality]
    if not member.isfile() or not member.name.startswith(f"{modality}/") or not member.name.endswith(suffix):
        raise RuntimeError(f"Unexpected official TAR member: {member.name}")
    return Path(member.name).name.removesuffix(suffix)


def main() -> None:
    assets = {item["path"]: item for item in json.loads((MANIFESTS / "impactmesh_assets.json").read_text(encoding="utf-8"))["assets"]}
    s1_path = verified_archive("S1RTC", assets)
    s2_path = verified_archive("S2L2A", assets)
    official = set((MANIFESTS / "impactmesh_flood_train.txt").read_text(encoding="utf-8").splitlines())
    if len(official) != 19448:
        raise RuntimeError("Official train split count changed")
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        roles = {row["activation"]: row["role"] for row in csv.DictReader(handle)}
    expected = {sample_id for sample_id in official if roles[sample_id.split("_", 1)[0]] == "train"}
    s1 = {}
    with tarfile.open(s1_path, "r:") as archive:
        for index, member in enumerate(archive, start=1):
            sample_id = member_id(member, "S1RTC")
            if sample_id not in official:
                raise RuntimeError(f"Unknown S1 member: {sample_id}")
            if sample_id not in expected:
                continue
            if sample_id in s1:
                raise RuntimeError(f"Unexpected or duplicate S1 member: {sample_id}")
            blob = archive.extractfile(member).read()
            group = zarr.open_consolidated(masked_store(blob, "S1RTC"), mode="r")
            times = group.attrs["datetime"].split(";")
            if len(times) != 4 or list(group["bands"].shape) != [4, 2, 256, 256]:
                raise RuntimeError(f"Unexpected S1 times or shape: {sample_id}")
            s1[sample_id] = (times[2],
                             CRS.from_wkt(group["spatial_ref"].attrs["crs_wkt"]),
                             np.asarray(group["x"][:]), np.asarray(group["y"][:]),
                             int(group["data_mask"][2]),
                             int(np.count_nonzero(group["nan_mask"][2])))
            if index % 1000 == 0:
                print(f"S1 archive members visited: {index}/{len(official)}; training role read: {len(s1)}", flush=True)
    if set(s1) != expected:
        raise RuntimeError("S1 archive does not match official train split")
    temp = OUTPUT.with_suffix(".csv.gz.part")
    seen = set()
    with gzip.open(temp, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        with tarfile.open(s2_path, "r:") as archive:
            for index, member in enumerate(archive, start=1):
                sample_id = member_id(member, "S2L2A")
                if sample_id not in official:
                    raise RuntimeError(f"Unknown S2 member: {sample_id}")
                if sample_id not in expected:
                    continue
                if sample_id not in s1 or sample_id in seen:
                    raise RuntimeError(f"Unexpected or duplicate S2 member: {sample_id}")
                seen.add(sample_id)
                blob = archive.extractfile(member).read()
                group = zarr.open_consolidated(masked_store(blob, "S2L2A"), mode="r")
                times = group.attrs["datetime"].split(";")
                if len(times) != 4 or list(group["bands"].shape) != [4, 12, 256, 256]:
                    raise RuntimeError(f"Unexpected S2 times or shape: {sample_id}")
                event_time, s1_crs, s1_x, s1_y, s1_data, s1_nan = s1[sample_id]
                s2_crs = CRS.from_wkt(group["spatial_ref"].attrs["crs_wkt"])
                grid_match = (s1_crs == s2_crs and
                              np.allclose(s1_x, group["x"][:], atol=1e-6, rtol=0) and
                              np.allclose(s1_y, group["y"][:], atol=1e-6, rtol=0))
                for time_index, value in enumerate(times):
                    scl = group["SCL"][time_index]
                    cloud = group["cloud_mask"][time_index]
                    nan = group["nan_mask"][time_index]
                    data = int(group["data_mask"][time_index])
                    if any(mask.shape != (256, 256) for mask in (scl, cloud, nan)) or data not in (0, 1):
                        raise RuntimeError(f"Unexpected optical mask: {sample_id}, {time_index}")
                    allowed_scl = np.isin(scl, (4, 5, 6))
                    cloud_clear = cloud == 0
                    nan_clear = nan == 0
                    upper = allowed_scl & cloud_clear & nan_clear & (data == 1)
                    writer.writerow({
                        "sample_id": sample_id, "activation": sample_id.split("_", 1)[0],
                        "time_index": time_index, "s1_event_utc": event_time, "s2_utc": value,
                        "s2_minus_s1_days": (utc(value) - utc(event_time)).total_seconds() / 86400,
                        "s2_at_or_before_s1": int(utc(value) <= utc(event_time)),
                        "s1_event_data_mask": s1_data, "s1_event_nan_pixels": s1_nan,
                        "s1_s2_grid_match": int(grid_match), "s2_data_mask": data,
                        "scl_4_5_6_pixels": int(np.count_nonzero(allowed_scl)),
                        "cloud_mask_0_pixels": int(np.count_nonzero(cloud_clear)),
                        "nan_mask_0_pixels": int(np.count_nonzero(nan_clear)),
                        "mask_joint_visible_upper_pixels": int(np.count_nonzero(upper)),
                        "mask_joint_visible_upper_fraction": float(np.count_nonzero(upper) / 65536),
                    })
                if index % 1000 == 0:
                    print(f"S2 archive members visited: {index}/{len(official)}; training role read: {len(seen)}", flush=True)
    if seen != expected:
        raise RuntimeError("S2 archive does not match official train split")
    temp.replace(OUTPUT)
    META.write_text(json.dumps({
        "source": "ImpactMesh-Flood v1 official train TARs, candidate training role only",
        "sample_count": len(expected), "rows": 4 * len(expected),
        "s1_tar_sha256": assets["train/S1RTC.tar"]["lfs"]["oid"],
        "s2_tar_sha256": assets["train/S2L2A.tar"]["lfs"]["oid"],
        "official_train_split_sha256": sha256(MANIFESTS / "impactmesh_flood_train.txt"),
        "candidate_process_split_sha256": sha256(SPLIT),
        "output_sha256": sha256(OUTPUT),
        "meaning": "Mask-only upper bound on visible optical pixels; S2 band finiteness/fill values not yet checked",
        "use_limit": "Candidate training-role audit only; no validation, calibration, internal or external test used; no eligibility threshold or final result",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Audited {len(expected)} training samples; wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
