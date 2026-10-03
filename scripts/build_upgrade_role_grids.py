"""Build candidate role-specific grid indexes without opening imagery or labels."""
from __future__ import annotations
import csv
import hashlib
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests"
ROLE_SOURCES = {
    "fit": ("impactmesh_train_mask_grid_index.csv", None),
    "calibration": ("impactmesh_train_mask_grid_index.csv", None),
    "internal_retro": ("impactmesh_train_mask_grid_index.csv", None),
    "holdout_retro": ("impactmesh_test_mask_grid_index.csv", "076f6cb08b7a76ca9de37fc3853225ae12099bba5a2b9005158c4d9ea5ab61ca"),
}
REQUIRED_GRID_FIELDS = {"sample_id", "grid_crs", "left", "bottom", "right", "top"}

def read(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=tuple(ROLE_SOURCES), action="append")
    args = parser.parse_args()
    for role in args.role or tuple(ROLE_SOURCES):
        source_name, expected_hash = ROLE_SOURCES[role]
        source = MANIFEST / source_name
        if expected_hash is not None and sha256(source) != expected_hash:
            raise RuntimeError(f"Grid source hash mismatch for {role}")
        source_rows = read(source)
        if not source_rows or not REQUIRED_GRID_FIELDS <= source_rows[0].keys():
            raise RuntimeError(f"Grid source lacks required geometry fields for {role}")
        grids = {row["sample_id"]: row for row in source_rows}
        if len(grids) != len(source_rows):
            raise RuntimeError(f"Grid source has duplicate sample IDs for {role}")
        rows = read(MANIFEST / f"g3_conservative_candidate_{role}.csv")
        if len({row["sample_id"] for row in rows}) != len(rows):
            raise RuntimeError(f"Candidate manifest has duplicate IDs for {role}")
        missing = [row["sample_id"] for row in rows if row["sample_id"] not in grids]
        if missing:
            raise RuntimeError(f"{role} has {len(missing)} IDs without grid metadata; first={missing[0]}")
        for row in (grids[item["sample_id"]] for item in rows):
            if not row["grid_crs"] or not all(row[key] for key in ("left", "bottom", "right", "top")):
                raise RuntimeError(f"Incomplete geometry for {role}: {row['sample_id']}")
        output = MANIFEST / f"upgrade_20260929_candidate_{role}_grids.csv"
        fields = list(next(iter(grids.values())).keys())
        with output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(grids[row["sample_id"]] for row in sorted(rows, key=lambda r: r["sample_id"]))

if __name__ == "__main__":
    main()
