"""Fetch one frozen train-pilot triplet from verified TAR member locations."""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

from rasterio.io import MemoryFile

from locate_impactmesh_tar import HEADER, HttpSource, SUFFIX, parse_header


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
OUT = ROOT / "data" / "raw" / "impactmesh" / "samples"
REPO = "ibm-esa-geospatial/ImpactMesh-Flood"
MANIFEST = MANIFESTS / "impactmesh_sample_manifest.csv"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-rank", type=int, required=True)
    args = parser.parse_args()
    with (MANIFESTS / "impactmesh_temporal_audit_frame.csv").open(newline="", encoding="utf-8") as handle:
        frame = {int(row["rank"]): row for row in csv.DictReader(handle)}
    if args.frame_rank not in frame:
        parser.error("Unknown frozen train-pilot rank")
    sample_id = frame[args.frame_rank]["sample_id"]
    ordinal = int(frame[args.frame_rank]["train_mask_tar_ordinal"])
    frozen = json.loads((MANIFESTS / "impactmesh_temporal_audit_frame_meta.json").read_text(encoding="utf-8"))
    index_path = MANIFESTS / "impactmesh_train_mask_index.csv"
    if hash_file(index_path) != frozen["source_index_sha256"]:
        raise RuntimeError("Verified train MASK index changed since pilot-frame selection")
    assets = {item["path"]: item for item in json.loads((MANIFESTS / "impactmesh_assets.json").read_text(encoding="utf-8"))["assets"]}
    with index_path.open(newline="", encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))
    mask_record = index[ordinal]
    if mask_record["sample_id"] != sample_id:
        raise RuntimeError("Frozen frame does not match verified train MASK TAR index")
    mask_tar = ROOT / "data" / "raw" / "impactmesh" / "train" / "MASK.tar"
    mask_asset = assets["train/MASK.tar"]
    if mask_tar.stat().st_size != mask_asset["size"] or hash_file(mask_tar) != mask_asset["lfs"]["oid"]:
        raise RuntimeError("Local train MASK TAR is not the verified official asset")

    rows = []
    for modality in ("MASK", "S1RTC", "S2L2A"):
        path = f"train/{modality}.tar"
        asset = assets[path]
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
        if modality == "MASK":
            offset = int(mask_record["member_offset_bytes"])
            size = int(mask_record["member_size_bytes"])
            with mask_tar.open("rb") as handle:
                handle.seek(offset - HEADER)
                header_bytes = handle.read(HEADER)
                content = handle.read(size)
            archive_verified = True
        else:
            location_path = MANIFESTS / f"impactmesh_tar_location_rank{args.frame_rank}_{modality}.json"
            location = json.loads(location_path.read_text(encoding="utf-8"))
            if location["frame_rank"] != args.frame_rank or location["sample_id"] != sample_id or location["tar_path"] != path or location["tar_lfs_sha256"] != asset["lfs"]["oid"]:
                raise RuntimeError(f"Untrusted TAR location: {location_path}")
            offset, size = int(location["member_offset_bytes"]), int(location["member_size_bytes"])
            source = HttpSource(url, asset["size"])
            header_bytes = source.read(offset - HEADER, HEADER)
            archive_verified = False
            content = source.read(offset, size)
        member = parse_header(header_bytes, offset - HEADER)
        expected_name = f"{modality}/{sample_id}{SUFFIX[modality]}"
        if member is None or member.name != expected_name or member.data_offset != offset or member.size != size or len(content) != size:
            raise RuntimeError(f"TAR header/payload mismatch for {path}: {sample_id}")
        if modality == "MASK":
            with MemoryFile(content) as memory:
                with memory.open() as raster:
                    if raster.width != 256 or raster.height != 256:
                        raise RuntimeError("Unexpected pilot label raster shape")
        else:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError(f"Corrupt Zarr ZIP member in {path}")
        target = OUT / sample_id / Path(member.name).name
        target.parent.mkdir(parents=True, exist_ok=True)
        content_hash = hashlib.sha256(content).hexdigest()
        if target.exists():
            if target.stat().st_size != size or hash_file(target) != content_hash:
                raise RuntimeError(f"Existing pilot file differs from official member: {target}")
        else:
            partial = target.with_suffix(target.suffix + ".part")
            partial.write_bytes(content)
            partial.replace(target)
        rows.append({
            "sample_id": sample_id,
            "split": "train",
            "tar_ordinal": ordinal,
            "modality": modality,
            "tar_path": path,
            "tar_source_url": url,
            "member_name": member.name,
            "member_offset_bytes": offset,
            "size_bytes": size,
            "local_path": target.relative_to(ROOT).as_posix(),
            "sha256": content_hash,
            "state": "sample-downloaded-tar-verified" if archive_verified else "sample-downloaded-tar-unverified",
        })
        print(f"{modality}: {member.name}, {size} bytes, SHA256 {content_hash}", flush=True)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        previous = [row for row in csv.DictReader(handle) if row["sample_id"] != sample_id]
    temp = MANIFEST.with_suffix(".csv.part")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(previous + rows)
    temp.replace(MANIFEST)
    print(f"Wrote three verified members to {MANIFEST}")


if __name__ == "__main__":
    main()
