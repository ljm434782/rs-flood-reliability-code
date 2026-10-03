"""Describe frozen pre-S1 optical ages without reading labels or predictions."""

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests"
OUT = ROOT / "outputs" / "formal_v1" / "summary_tables" / "selected_s2_age.csv"


def rows(path, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def main():
    split = {row["activation"]: row for row in rows(MANIFEST / "impactmesh_process_split_grouped.csv")}
    exclusions = {row["activation"] for row in rows(
        MANIFEST / "impactmesh_evaluation_process_exclusions.csv")}
    groups = defaultdict(list)
    for row in rows(MANIFEST / "impactmesh_all_train_frozen_eligibility.csv.gz", True):
        if row["paired_eligible"] != "1" or row["activation"] in exclusions:
            continue
        role = split[row["activation"]]["role"]
        if role in {"train", "validation", "calibration", "internal_test"}:
            groups[role].append(float(row["selected_age_days"]))
    holdout_exclusions = {"EMSR757", "EMSR763", "EMSR766"}
    by_process = defaultdict(list)
    for row in rows(MANIFEST / "impactmesh_holdout_frozen_eligibility.csv"):
        if row["paired_eligible"] == "1" and row["activation"] not in holdout_exclusions:
            by_process[row["activation"]].append(float(row["selected_age_days"]))
    for ages in by_process.values():
        if len(ages) >= 5:
            groups["official_holdout"].extend(ages)
    output = []
    for role in ("train", "validation", "calibration", "internal_test", "official_holdout"):
        values = np.asarray(groups[role])
        if not len(values) or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 30)):
            raise RuntimeError(f"Invalid frozen S2 ages for {role}")
        output.append({"role": role, "tiles": len(values), "min_days": float(values.min()),
                       "q25_days": float(np.quantile(values, .25)),
                       "median_days": float(np.median(values)),
                       "q75_days": float(np.quantile(values, .75)),
                       "max_days": float(values.max())})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(output, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
