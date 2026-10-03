"""Compute full-band optical validity for candidate training-role samples only."""

import csv
import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
ARCHIVE = ROOT / "data" / "raw" / "impactmesh" / "train" / "S2L2A.tar"
MASK_INPUT = MANIFESTS / "impactmesh_candidate_train_s2_availability.csv.gz"
MASK_META = MANIFESTS / "impactmesh_candidate_train_s2_availability_meta.json"
OUTPUT = MANIFESTS / "impactmesh_candidate_train_optical_full.csv.gz"
META = MANIFESTS / "impactmesh_candidate_train_optical_full_meta.json"
SUFFIX = "_S2L2A.zarr.zip"
FIELDS = ["sample_id", "activation", "time_index", "s1_event_utc", "s2_utc",
          "s2_minus_s1_days", "s2_at_or_before_s1", "s1_event_data_mask",
          "s1_event_nan_pixels", "s1_s2_grid_match", "s2_data_mask",
          "scl_4_5_6_pixels", "cloud_mask_0_pixels", "nan_mask_0_pixels",
          "all_12_bands_finite_nonfill_pixels", "joint_visible_pixels",
          "joint_visible_fraction"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_store(blob: bytes) -> zarr.storage.MemoryStore:
    store = zarr.storage.MemoryStore()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in archive.namelist():
            store[name] = archive.read(name)
    return store


def main() -> None:
    mask_meta = json.loads(MASK_META.read_text(encoding="utf-8"))
    if sha256(MASK_INPUT) != mask_meta["output_sha256"]:
        raise RuntimeError("Mask-only audit differs from its metadata")
    assets = {item["path"]: item for item in json.loads(
        (MANIFESTS / "impactmesh_assets.json").read_text(encoding="utf-8"))["assets"]}
    asset = assets["train/S2L2A.tar"]
    if ARCHIVE.stat().st_size != int(asset["size"]) or sha256(ARCHIVE) != asset["lfs"]["oid"]:
        raise RuntimeError("Official S2 archive has not passed SHA256")

    prior = {}
    with gzip.open(MASK_INPUT, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["sample_id"], int(row["time_index"]))
            if key in prior:
                raise RuntimeError(f"Duplicate mask audit row: {key}")
            prior[key] = row
    expected = {sample_id for sample_id, _ in prior}
    if len(expected) != int(mask_meta["sample_count"]) or len(prior) != int(mask_meta["rows"]):
        raise RuntimeError("Mask-only audit count mismatch")

    temp = OUTPUT.with_suffix(".csv.gz.part")
    seen = set()
    rows_written = 0
    with gzip.open(temp, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        with tarfile.open(ARCHIVE, "r:") as archive:
            for index, member in enumerate(archive, start=1):
                if not member.isfile() or not member.name.startswith("S2L2A/") or not member.name.endswith(SUFFIX):
                    raise RuntimeError(f"Unexpected official S2 member: {member.name}")
                sample_id = Path(member.name).name.removesuffix(SUFFIX)
                if sample_id not in expected:
                    continue
                if sample_id in seen:
                    raise RuntimeError(f"Duplicate S2 member: {sample_id}")
                seen.add(sample_id)
                blob = archive.extractfile(member).read()
                group = zarr.open_consolidated(full_store(blob), mode="r")
                if list(group["bands"].shape) != [4, 12, 256, 256]:
                    raise RuntimeError(f"Unexpected S2 shape: {sample_id}")
                for time_index in range(4):
                    source = prior[(sample_id, time_index)]
                    scl = np.asarray(group["SCL"][time_index])
                    cloud = np.asarray(group["cloud_mask"][time_index])
                    nan = np.asarray(group["nan_mask"][time_index])
                    bands = np.asarray(group["bands"][time_index])
                    data = int(group["data_mask"][time_index])
                    allowed_scl = np.isin(scl, (4, 5, 6))
                    cloud_clear = cloud == 0
                    nan_clear = nan == 0
                    finite_nonfill = np.all(np.isfinite(bands) & (bands != -9999), axis=0)
                    joint = allowed_scl & cloud_clear & nan_clear & finite_nonfill & (data == 1)
                    checks = {
                        "s2_data_mask": data,
                        "scl_4_5_6_pixels": int(np.count_nonzero(allowed_scl)),
                        "cloud_mask_0_pixels": int(np.count_nonzero(cloud_clear)),
                        "nan_mask_0_pixels": int(np.count_nonzero(nan_clear)),
                    }
                    if any(int(source[name]) != value for name, value in checks.items()):
                        raise RuntimeError(f"Mask audit mismatch: {sample_id}, {time_index}")
                    writer.writerow({
                        **{name: source[name] for name in FIELDS[:14]},
                        "all_12_bands_finite_nonfill_pixels": int(np.count_nonzero(finite_nonfill)),
                        "joint_visible_pixels": int(np.count_nonzero(joint)),
                        "joint_visible_fraction": float(np.count_nonzero(joint) / 65536),
                    })
                    rows_written += 1
                if index % 1000 == 0:
                    print(f"S2 archive members visited: {index}/19448; full-band training samples: {len(seen)}", flush=True)
    if seen != expected or rows_written != len(prior):
        raise RuntimeError("Full-band scan does not match mask-only training-role audit")
    temp.replace(OUTPUT)
    META.write_text(json.dumps({
        "source": "ImpactMesh-Flood v1 official train/S2L2A.tar, candidate training role only",
        "sample_count": len(seen),
        "rows": rows_written,
        "s2_tar_sha256": asset["lfs"]["oid"],
        "mask_input_sha256": sha256(MASK_INPUT),
        "output_sha256": sha256(OUTPUT),
        "joint_visible_definition": "SCL in {4,5,6} AND cloud_mask==0 AND nan_mask==0 AND all 12 bands finite and !=-9999 AND data_mask==1",
        "use_limit": "Candidate training-role audit only; no held-out role read and no eligibility threshold selected",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Full-band audited {len(seen)} training samples; wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
