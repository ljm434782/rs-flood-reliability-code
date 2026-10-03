"""Fetch a matched ImpactMesh triplet by TAR ordinal via HTTP ranges."""

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

from index_sturm_zip import RangeReader


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data" / "manifests" / "impactmesh_assets.json"
OUT = ROOT / "data" / "raw" / "impactmesh" / "samples"
MANIFEST = ROOT / "data" / "manifests" / "impactmesh_sample_manifest.csv"
REPO = "ibm-esa-geospatial/ImpactMesh-Flood"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--ordinal", type=int, help="Zero-based member position in each modality TAR")
    parser.add_argument("--frame-rank", type=int, help="Rank in the frozen train temporal-audit frame")
    args = parser.parse_args()
    split = args.split
    expected_frame_id = None
    if args.frame_rank is not None:
        if args.ordinal is not None or split != "train":
            parser.error("--frame-rank requires --split train and no --ordinal")
        with (ROOT / "data" / "manifests" / "impactmesh_temporal_audit_frame.csv").open(newline="", encoding="utf-8") as handle:
            frame = {int(row["rank"]): row for row in csv.DictReader(handle)}
        if args.frame_rank not in frame:
            parser.error("--frame-rank is absent from the frozen frame")
        args.ordinal = int(frame[args.frame_rank]["train_mask_tar_ordinal"])
        expected_frame_id = frame[args.frame_rank]["sample_id"]
    if args.ordinal is None:
        args.ordinal = 0
    if args.ordinal < 0:
        parser.error("--ordinal must be nonnegative")
    if args.ordinal > 256:
        parser.error("This sequential downloader is limited to ordinal <=256; use a verified random-access locator for later members")
    assets = {item["path"]: item for item in json.loads(ASSETS.read_text(encoding="utf-8"))["assets"]}
    expected_id = None
    rows = []
    for modality, suffix in (("MASK", "_annotation_flood.tif"), ("S1RTC", "_S1RTC.zarr.zip"), ("S2L2A", "_S2L2A.zarr.zip")):
        path = f"{split}/{modality}.tar"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
        local_tar = ROOT / "data" / "raw" / "impactmesh" / split / "MASK.tar"
        if modality == "MASK" and local_tar.exists():
            if local_tar.stat().st_size != assets[path]["size"] or hashlib.sha256(local_tar.read_bytes()).hexdigest() != assets[path]["lfs"]["oid"]:
                raise RuntimeError(f"Local MASK TAR does not match official asset: {local_tar}")
        archive_source = tarfile.open(local_tar, mode="r:") if modality == "MASK" and local_tar.exists() else tarfile.open(
            fileobj=RangeReader(url, assets[path]["size"], block_size=64 * 1024), mode="r:"
        )
        with archive_source as archive:
            member = None
            for _ in range(args.ordinal + 1):
                member = archive.next()
            if member is None or not member.isfile() or not member.name.endswith(suffix):
                raise RuntimeError(f"Unexpected member at ordinal {args.ordinal} in {path}")
            sample_id = Path(member.name).name.removesuffix(suffix)
            if expected_id is None:
                expected_id = sample_id
            elif sample_id != expected_id:
                raise RuntimeError(f"Mismatched {split} sample at ordinal {args.ordinal}: {sample_id} != {expected_id}")
            if expected_frame_id is not None and sample_id != expected_frame_id:
                raise RuntimeError(f"Frozen frame and {path} disagree at ordinal {args.ordinal}: {sample_id} != {expected_frame_id}")
            target = OUT / sample_id / Path(member.name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Cannot read {member.name}")
                target.write_bytes(source.read())
            if target.stat().st_size != member.size:
                raise RuntimeError(f"Incorrect size for {target}")
            rows.append({
                "sample_id": sample_id,
                "split": split,
                "tar_ordinal": args.ordinal,
                "modality": modality,
                "tar_path": path,
                "tar_source_url": url,
                "member_name": member.name,
                "member_offset_bytes": member.offset_data,
                "size_bytes": member.size,
                "local_path": target.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "state": "sample-downloaded-tar-unverified",
            })
            print(f"{modality}: {target.name} ({member.size} bytes)", flush=True)
    official_split = (ROOT / "data" / "manifests" / f"impactmesh_flood_{split}.txt").read_text(encoding="utf-8").splitlines()
    if expected_id not in official_split:
        raise RuntimeError(f"Sample {expected_id} absent from official {split} split")
    if split == "test":
        holdout = (ROOT / "data" / "manifests" / "impactmesh_flood_test_holdout.txt").read_text(encoding="utf-8").splitlines()
        if expected_id in holdout:
            for row in rows:
                row["split"] = "test_holdout"
    if MANIFEST.exists():
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            previous = list(csv.DictReader(handle))
        previous = [row for row in previous if row["sample_id"] != expected_id]
        for row in previous:
            row.setdefault("split", "val")
            row.setdefault("tar_ordinal", 0)
    else:
        previous = []
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(previous + rows)
    print(f"Official {rows[0]['split']} split contains {expected_id}; wrote {MANIFEST}")


if __name__ == "__main__":
    main()
