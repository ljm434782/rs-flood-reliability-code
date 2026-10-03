"""Read georeferencing headers of official ImpactMesh mask TAR members.

No label pixels are read. The TIFF grid, not the sample ID's MGRS token or
the source-image proj_code attribute, is authoritative for each mask tile.
"""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import rasterio
from rasterio.io import MemoryFile


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
ASSETS = MANIFESTS / "impactmesh_assets.json"
ID = re.compile(r"^EMSR\d+_\d+_(\d{2})([C-HJ-NP-X])[A-Z]{2}_x(\d+)_y(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_zone_crs(sample_id: str) -> str:
    match = ID.fullmatch(sample_id)
    if not match:
        raise RuntimeError(f"Unexpected sample ID: {sample_id}")
    zone = int(match.group(1))
    return f"EPSG:{(32600 if match.group(2) >= 'N' else 32700) + zone}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    args = parser.parse_args()
    split = args.split
    tar = ROOT / "data" / "raw" / "impactmesh" / split / "MASK.tar"
    index = MANIFESTS / f"impactmesh_{split}_mask_index.csv"
    assets = {item["path"]: item for item in json.loads(ASSETS.read_text(encoding="utf-8"))["assets"]}
    published = assets[f"{split}/MASK.tar"]
    if tar.stat().st_size != published["size"] or sha256(tar) != published["lfs"]["oid"]:
        raise RuntimeError("Mask TAR differs from published size or SHA256")
    with index.open(newline="", encoding="utf-8") as handle:
        indexed = list(csv.DictReader(handle))
    if len(indexed) != len({row["sample_id"] for row in indexed}):
        raise RuntimeError("Duplicate sample ID in mask index")
    rows = []
    with tar.open("rb") as handle:
        for item in indexed:
            sample_id = item["sample_id"]
            handle.seek(int(item["member_offset_bytes"]))
            payload = handle.read(int(item["member_size_bytes"]))
            if len(payload) != int(item["member_size_bytes"]):
                raise RuntimeError(f"Truncated mask member: {sample_id}")
            with MemoryFile(payload) as memory, memory.open() as raster:
                if raster.count != 1 or raster.width != 256 or raster.height != 256 or raster.crs is None:
                    raise RuntimeError(f"Unexpected mask grid: {sample_id}")
                affine = raster.transform
                if affine.b != 0 or affine.d != 0 or affine.a != 10 or affine.e != -10:
                    raise RuntimeError(f"Unexpected mask transform: {sample_id}")
                bounds = raster.bounds
                match = ID.fullmatch(sample_id)
                if not match or abs(int(match.group(3)) - (bounds.left + 5)) > 1e-6 or abs(int(match.group(4)) - (bounds.top - 5)) > 1e-6:
                    raise RuntimeError(f"Sample ID pixel-center coordinates disagree with TIFF header: {sample_id}")
                grid_crs = str(raster.crs)
                rows.append({
                    "split": split,
                    "sample_id": sample_id,
                    "activation": item["activation"],
                    "holdout": item["holdout"],
                    "grid_crs": grid_crs,
                    "id_zone_crs": id_zone_crs(sample_id),
                    "id_zone_matches_grid": int(grid_crs == id_zone_crs(sample_id)),
                    "left": bounds.left,
                    "bottom": bounds.bottom,
                    "right": bounds.right,
                    "top": bounds.top,
                })
    output = MANIFESTS / f"impactmesh_{split}_mask_grid_index.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mismatches = [row for row in rows if not row["id_zone_matches_grid"]]
    meta = {
        "split": split,
        "source_tar_sha256": published["lfs"]["oid"],
        "source_index_sha256": sha256(index),
        "grid_index_sha256": sha256(output),
        "member_count": len(rows),
        "id_zone_crs_mismatch_count": len(mismatches),
        "id_zone_crs_mismatch_examples": [row["sample_id"] for row in mismatches[:20]],
        "header_only": True,
        "scope": "Grid CRS and bounds from each TIFF header; no label pixel values read",
    }
    meta_path = MANIFESTS / f"impactmesh_{split}_mask_grid_index_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"{split}: {len(rows)} headers; {len(mismatches)} ID-zone/grid-CRS mismatches; wrote {output}")


if __name__ == "__main__":
    main()
