"""Recompute G3 candidate-manifest counts from IDs and role fields only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests"
ROLE_FILES = {
    "fit": "g3_conservative_candidate_fit.csv",
    "calibration": "g3_conservative_candidate_calibration.csv",
    "internal_retro": "g3_conservative_candidate_internal_retro.csv",
    "holdout_retro": "g3_conservative_candidate_holdout_retro.csv",
    "historical_dev_inaccessible": "g3_conservative_candidate_historical_dev_inaccessible.csv",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {role: read_rows(MANIFEST / filename) for role, filename in ROLE_FILES.items()}
    ids = {role: {row["sample_id"] for row in values} for role, values in rows.items()}
    if any(len(ids[role]) != len(values) for role, values in rows.items()):
        raise RuntimeError("Candidate manifest has duplicate sample IDs within a role")
    role_overlap = {
        f"{left}__{right}": len(ids[left] & ids[right])
        for index, left in enumerate(ids) for right in list(ids)[index + 1:]
    }
    if any(role_overlap.values()):
        raise RuntimeError(f"Cross-role sample overlap: {role_overlap}")
    fit = rows["fit"]
    unit_sizes = Counter(row["conservative_unit_id"] for row in fit)
    if len(unit_sizes) != 60:
        raise RuntimeError(f"Expected 60 units, found {len(unit_sizes)}")
    if [int(row["global_order"]) for row in fit] != list(range(1, len(fit) + 1)):
        raise RuntimeError("Fit global_order is not a complete consecutive sequence")
    for unit in unit_sizes:
        unit_rows = [row for row in fit if row["conservative_unit_id"] == unit]
        expected = list(range(1, len(unit_rows) + 1))
        actual = sorted(int(row["within_unit_order"]) for row in unit_rows)
        if actual != expected:
            raise RuntimeError(f"Non-consecutive within-unit order for {unit}")
    draws_per_unit = 24 * 8
    unique_draws = sum(min(size, draws_per_unit) for size in unit_sizes.values())
    report = {
        "scope": "candidate-only IDs/roles/orders; no imagery, labels, predictions, scores, or training",
        "candidate_status": json.loads((MANIFEST / "g3_conservative_candidate_meta.json").read_text(encoding="utf-8"))["status"],
        "file_sha256": {name: file_hash(MANIFEST / filename) for name, filename in ROLE_FILES.items()},
        "role_sample_counts": {role: len(values) for role, values in ids.items()},
        "cross_role_sample_overlap": role_overlap,
        "fit": {
            "units": len(unit_sizes),
            "sample_ids": len(ids["fit"]),
            "epochs": 24,
            "tiles_per_unit_per_epoch": 8,
            "batch_size": 4,
            "draws_per_run": len(unit_sizes) * draws_per_unit,
            "updates_per_run": len(unit_sizes) * draws_per_unit // 4,
            "unique_tile_draws_per_run": unique_draws,
            "unique_tile_coverage": unique_draws / len(ids["fit"]),
            "unit_size_min": min(unit_sizes.values()),
            "unit_size_max": max(unit_sizes.values()),
            "units_exhausted_at_192": sum(size <= draws_per_unit for size in unit_sizes.values()),
            "units_repeated_after_192": sum(size > draws_per_unit for size in unit_sizes.values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
