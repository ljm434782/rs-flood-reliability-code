"""Verify one official ImpactMesh mask TAR and index names without reading labels."""

import argparse
import csv
import hashlib
import json
import tarfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data" / "manifests" / "impactmesh_assets.json"
SUFFIX = "_annotation_flood.tif"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    args = parser.parse_args()
    split = args.split
    target = ROOT / "data" / "raw" / "impactmesh" / split / "MASK.tar"
    index = ROOT / "data" / "manifests" / f"impactmesh_{split}_mask_index.csv"
    meta = ROOT / "data" / "manifests" / f"impactmesh_{split}_mask_index_meta.json"
    url = f"https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Flood/resolve/main/{split}/MASK.tar"
    assets = {item["path"]: item for item in json.loads(ASSETS.read_text(encoding="utf-8"))["assets"]}
    expected = assets[f"{split}/MASK.tar"]["lfs"]["oid"]
    expected_size = assets[f"{split}/MASK.tar"]["size"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(".tar.part")
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, connect=3, read=3, backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
        )))
        with session.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    handle.write(chunk)
        partial.replace(target)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest = digest.hexdigest()
    if target.stat().st_size != expected_size or digest != expected:
        raise RuntimeError(f"Official TAR size or SHA256 mismatch: {target}")

    split_ids = set((ROOT / "data" / "manifests" / f"impactmesh_flood_{split}.txt").read_text(encoding="utf-8").splitlines())
    holdout_ids = set((ROOT / "data" / "manifests" / "impactmesh_flood_test_holdout.txt").read_text(encoding="utf-8").splitlines()) if split == "test" else set()
    rows = []
    with tarfile.open(target, "r:") as archive:
        for ordinal, member in enumerate(archive):
            if not member.isfile() or not member.name.endswith(SUFFIX):
                raise RuntimeError(f"Unexpected member: {member.name}")
            sample_id = Path(member.name).name.removesuffix(SUFFIX)
            if sample_id not in split_ids:
                raise RuntimeError(f"{split} TAR member absent from official split: {sample_id}")
            rows.append({
                "ordinal": ordinal,
                "sample_id": sample_id,
                "activation": sample_id.split("_")[0],
                "holdout": int(sample_id in holdout_ids),
                "member_offset_bytes": member.offset_data,
                "member_size_bytes": member.size,
            })
    if len(rows) != len(split_ids) or {row["sample_id"] for row in rows} != split_ids:
        raise RuntimeError(f"{split} mask TAR and split sample sets differ")
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "source_url": url,
        "lfs_sha256": expected,
        "downloaded_sha256": digest,
        "size_bytes": expected_size,
        "member_count": len(rows),
        "activation_count": len({row["activation"] for row in rows}),
        "selection": "all mask member names indexed; label pixels not read",
    }
    if split == "test":
        first_holdout = next(row for row in rows if row["holdout"])
        report.update({
            "first_holdout_ordinal": first_holdout["ordinal"],
            "first_holdout_sample_id": first_holdout["sample_id"],
            "selection": "first test TAR member listed in official test_holdout.txt; index only, no label pixels read",
        })
    meta.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Verified {split}/MASK.tar: size={expected_size}, SHA256={digest}")
    print(f"Indexed {len(rows)} {split} masks across {report['activation_count']} activations")


if __name__ == "__main__":
    main()
