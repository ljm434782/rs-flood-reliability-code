"""Freeze a fresh process-disjoint v4 fit/development split using IDs only."""

import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

from scripts.train_impactmesh_formal import ELIGIBILITY, ROOT, SPLIT, sha256

CONFIG = ROOT / "configs/impactmesh_exploratory_v4.json"
OUT = ROOT / "data/manifests/impactmesh_v4_training_dev_processes.csv"
META = OUT.with_suffix(".json")
MISSING = {"visible_below_0.5", "no_pre_s1_s2_within_30d"}


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        activations = {row["activation"]: row for row in csv.DictReader(handle)}
    ids = {row["process_id"] for row in activations.values() if row["role"] == "train"}
    if len(ids) != 81:
        raise RuntimeError("Original train process count changed")
    hash_for = lambda process: hashlib.sha256((config["dev_process_hash_seed"] + "|" +
                                               process.lower()).encode("ascii")).hexdigest()
    ordered = sorted(ids, key=hash_for)
    dev = set(ordered[:config["dev_process_count"]])
    paired, missing = Counter(), Counter()
    with gzip.open(ELIGIBILITY, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = activations[row["activation"]]
            if group["role"] != "train":
                continue
            process = group["process_id"]
            if row["paired_eligible"] == "1":
                paired[process] += 1
            elif row["failure_reasons"] in MISSING:
                missing[process] += 1
    rows = [{"process_id": process, "v4_role": "dev" if process in dev else "fit",
             "paired_eligible_tiles": paired[process], "unqualified_s2_tiles": missing[process],
             "split_hash": hash_for(process)} for process in sorted(ids)]
    if any(row["paired_eligible_tiles"] < 5 for row in rows):
        raise RuntimeError("A process fell below frozen five-pair minimum")
    if OUT.exists() or META.exists():
        raise RuntimeError("Refusing to overwrite v4 split")
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    META.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "hash-only fresh split of original training process IDs; no label pixels read",
        "config_sha256": sha256(CONFIG), "original_split_sha256": sha256(SPLIT),
        "eligibility_sha256": sha256(ELIGIBILITY), "output_sha256": sha256(OUT),
        "fit_processes": len(ids)-len(dev), "dev_processes": len(dev),
        "fit_paired_tiles": sum(paired[p] for p in ids if p not in dev),
        "dev_paired_tiles": sum(paired[p] for p in dev),
        "dev_real_missing_processes_at_least_two": sum(missing[p] >= 2 for p in dev),
        "dev_real_missing_tiles": sum(missing[p] for p in dev),
        "test_labels_or_metrics_read": False}, indent=2) + "\n", encoding="utf-8")
    print(META.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
