"""Inventory the verified train pilot ZIP layout and published archive sizes."""

import csv
import hashlib
import json
import statistics
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
OUTPUT = MANIFESTS / "impactmesh_acquisition_feasibility.json"
MASK_NAMES = {"SCL/0.0.0", "cloud_mask/0.0.0", "nan_mask/0.0.0", "data_mask/0"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: list[int]) -> dict[str, int | float]:
    return {"min": min(values), "median": statistics.median(values), "max": max(values), "sum": sum(values)}


def main() -> None:
    frame_path = MANIFESTS / "impactmesh_temporal_audit_frame.csv"
    with frame_path.open(newline="", encoding="utf-8") as handle:
        frame = list(csv.DictReader(handle))
    if len(frame) != 24 or len({row["activation"] for row in frame}) != 24:
        raise RuntimeError("Unexpected pilot frame")
    if sha(frame_path) != json.loads((MANIFESTS / "impactmesh_optical_joint_audit_meta.json").read_text(encoding="utf-8"))["frozen_frame_sha256"]:
        raise RuntimeError("Pilot frame hash differs from optical audit")
    with (MANIFESTS / "impactmesh_sample_manifest.csv").open(newline="", encoding="utf-8") as handle:
        member_rows = {(row["sample_id"], row["modality"]): row for row in csv.DictReader(handle)
                       if row["split"] == "train" and row["modality"] == "S2L2A"}
    if not {row["sample_id"] for row in frame} <= {key[0] for key in member_rows}:
        raise RuntimeError("Pilot S2 member missing from source manifest")
    member_bytes, mask_bytes, metadata_bytes, band_bytes = [], [], [], []
    for row in frame:
        source = member_rows[(row["sample_id"], "S2L2A")]
        path = ROOT / source["local_path"]
        if path.stat().st_size != int(source["size_bytes"]) or sha(path) != source["sha256"]:
            raise RuntimeError(f"S2 ZIP differs from source manifest: {path}")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            by_name = {info.filename: info for info in infos}
            if (len(infos) != 33 or len(by_name) != 33 or
                    not MASK_NAMES <= set(by_name) or ".zmetadata" not in by_name or
                    "bands/0.0.0.0" not in by_name or
                    any(info.compress_type != zipfile.ZIP_STORED for info in infos)):
                raise RuntimeError(f"Unexpected S2 ZIP layout: {path}")
            member_bytes.append(path.stat().st_size)
            mask_bytes.append(sum(by_name[name].file_size for name in MASK_NAMES))
            metadata_bytes.append(by_name[".zmetadata"].file_size)
            band_bytes.append(by_name["bands/0.0.0.0"].file_size)
    index_path = MANIFESTS / "impactmesh_train_mask_index.csv"
    index_meta = json.loads((MANIFESTS / "impactmesh_train_mask_index_meta.json").read_text(encoding="utf-8"))
    with index_path.open(newline="", encoding="utf-8") as handle:
        counts = Counter(row["activation"] for row in csv.DictReader(handle))
    if sum(counts.values()) != index_meta["member_count"] or len(counts) != index_meta["activation_count"]:
        raise RuntimeError("Train MASK index population changed")
    assets_path = MANIFESTS / "impactmesh_assets.json"
    assets = {item["path"]: item for item in json.loads(assets_path.read_text(encoding="utf-8"))["assets"]}
    mask_tar = ROOT / "data" / "raw" / "impactmesh" / "train" / "MASK.tar"
    if (sha(mask_tar) != index_meta["downloaded_sha256"] or
            index_meta["downloaded_sha256"] != index_meta["lfs_sha256"] or
            index_meta["lfs_sha256"] != assets["train/MASK.tar"]["lfs"]["oid"]):
        raise RuntimeError("Local train MASK TAR differs from publisher record")
    result = {
        "scope": "asset inventory and train-only pilot ZIP layout; not optical eligibility or process qualification",
        "pilot_frame_sha256": sha(frame_path),
        "train_mask_index_sha256": sha(index_path),
        "train_mask_tar_verified_sha256": index_meta["downloaded_sha256"],
        "asset_manifest_sha256": sha(assets_path),
        "train_archives": {name: {"bytes": assets[name]["size"], "publisher_lfs_sha256": assets[name]["lfs"]["oid"]}
                           for name in ("train/MASK.tar", "train/S1RTC.tar", "train/S2L2A.tar")},
        "train_activation_count": len(counts),
        "train_chip_count": sum(counts.values()),
        "chips_per_activation": {"min": min(counts.values()), "median": statistics.median(counts.values()),
                                 "max": max(counts.values())},
        "activations_with_at_least_n_chips": {str(n): sum(v >= n for v in counts.values())
                                             for n in (1, 2, 5, 10, 20, 50, 100)},
        "verified_pilot_s2_zip_count": len(member_bytes),
        "pilot_s2_zip_member_bytes": stats(member_bytes),
        "pilot_s2_mask_chunk_bytes": stats(mask_bytes),
        "pilot_s2_consolidated_metadata_bytes": stats(metadata_bytes),
        "pilot_s2_band_chunk_bytes": stats(band_bytes),
        "limits": [
            "ZIP chunks are small locally, but locating every member inside remote TAR still needs an index or remote reads.",
            "SCL/cloud/nan/data chunks omit the 12-band finite/nonfill condition, so masks alone cannot certify the joint-visible definition.",
            "Full S1RTC/S2L2A TAR publisher hashes and all-process optical/label quality remain unverified.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
