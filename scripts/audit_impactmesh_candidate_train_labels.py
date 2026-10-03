"""Audit label codes and class counts for candidate training-role samples only."""

import csv
import gzip
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
ARCHIVE = ROOT / "data" / "raw" / "impactmesh" / "train" / "MASK.tar"
OPTICAL = MANIFESTS / "impactmesh_candidate_train_optical_full.csv.gz"
OPTICAL_META = MANIFESTS / "impactmesh_candidate_train_optical_full_meta.json"
OUTPUT = MANIFESTS / "impactmesh_candidate_train_label_counts.csv.gz"
META = MANIFESTS / "impactmesh_candidate_train_label_counts_meta.json"
SUFFIX = "_annotation_flood.tif"
FIELDS = ["sample_id", "activation", "ignore_pixels", "nonwater_pixels",
          "permanent_water_pixels", "flood_pixels", "supervised_pixels",
          "permanent_water_present", "flood_present"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    optical_meta = json.loads(OPTICAL_META.read_text(encoding="utf-8"))
    if sha256(OPTICAL) != optical_meta["output_sha256"]:
        raise RuntimeError("Full optical audit differs from its metadata")
    expected = set()
    with gzip.open(OPTICAL, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            expected.add(row["sample_id"])
    if len(expected) != int(optical_meta["sample_count"]):
        raise RuntimeError("Full optical audit sample count mismatch")

    assets = {item["path"]: item for item in json.loads(
        (MANIFESTS / "impactmesh_assets.json").read_text(encoding="utf-8"))["assets"]}
    asset = assets["train/MASK.tar"]
    if ARCHIVE.stat().st_size != int(asset["size"]) or sha256(ARCHIVE) != asset["lfs"]["oid"]:
        raise RuntimeError("Official train label archive has not passed SHA256")

    rows = []
    seen = set()
    class_totals = Counter()
    with tarfile.open(ARCHIVE, "r:") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(SUFFIX):
                raise RuntimeError(f"Unexpected label member: {member.name}")
            sample_id = Path(member.name).name.removesuffix(SUFFIX)
            if sample_id not in expected:
                continue
            if sample_id in seen:
                raise RuntimeError(f"Duplicate label member: {sample_id}")
            seen.add(sample_id)
            blob = archive.extractfile(member).read()
            with MemoryFile(blob) as memory, memory.open() as source:
                label = source.read(1)
                if source.shape != (256, 256) or source.count != 1:
                    raise RuntimeError(f"Unexpected label shape/count: {sample_id}")
            values = set(map(int, np.unique(label)))
            if not values <= {-1, 0, 1, 2}:
                raise RuntimeError(f"Unexpected label codes {values}: {sample_id}")
            counts = {value: int(np.count_nonzero(label == value)) for value in (-1, 0, 1, 2)}
            if sum(counts.values()) != 65536:
                raise RuntimeError(f"Label pixel count mismatch: {sample_id}")
            for value, count in counts.items():
                class_totals[str(value)] += count
            rows.append({
                "sample_id": sample_id,
                "activation": sample_id.split("_", 1)[0],
                "ignore_pixels": counts[-1],
                "nonwater_pixels": counts[0],
                "permanent_water_pixels": counts[1],
                "flood_pixels": counts[2],
                "supervised_pixels": counts[0] + counts[1] + counts[2],
                "permanent_water_present": int(counts[1] > 0),
                "flood_present": int(counts[2] > 0),
            })
    if seen != expected:
        raise RuntimeError("Candidate training-role labels do not match optical audit")
    rows.sort(key=lambda row: row["sample_id"])
    temp = OUTPUT.with_suffix(".csv.gz.part")
    with gzip.open(temp, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(OUTPUT)
    process_flood = {row["activation"] for row in rows if row["flood_present"]}
    META.write_text(json.dumps({
        "source": "ImpactMesh-Flood v1 official train/MASK.tar, candidate training role only",
        "sample_count": len(rows),
        "process_count": len({row["activation"] for row in rows}),
        "samples_with_permanent_water": sum(row["permanent_water_present"] for row in rows),
        "samples_with_flood": sum(row["flood_present"] for row in rows),
        "processes_with_flood": len(process_flood),
        "class_pixel_totals": dict(class_totals),
        "mask_tar_sha256": asset["lfs"]["oid"],
        "optical_input_sha256": sha256(OPTICAL),
        "output_sha256": sha256(OUTPUT),
        "use_limit": "Candidate training role only; descriptive audit, not a rule for excluding no-flood chips",
    }, indent=2) + "\n", encoding="utf-8")
    print(META.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
