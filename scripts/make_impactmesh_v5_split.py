"""Freeze v5 train-only fit/development split using process IDs, without labels."""

import csv
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

from scripts.train_impactmesh_formal import ELIGIBILITY, ROOT, SPLIT, sha256

CONFIG = ROOT / "configs/impactmesh_exploratory_v5.json"
V4_SPLIT = ROOT / "data/manifests/impactmesh_v4_training_dev_processes.csv"
OUT = ROOT / "data/manifests/impactmesh_v5_training_dev_processes.csv"
META = OUT.with_suffix(".json")
MISSING = {"visible_below_0.5", "no_pre_s1_s2_within_30d"}


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        activations = {row["activation"]: row for row in csv.DictReader(handle)}
    ids = {row["process_id"] for row in activations.values() if row["role"] == "train"}
    with V4_SPLIT.open(newline="", encoding="utf-8") as handle:
        old = {row["process_id"]: row for row in csv.DictReader(handle)}
    if len(ids) != 81 or set(old) != ids or sum(r["v4_role"] == "fit" for r in old.values()) != 65:
        raise RuntimeError("Original training or v4 split changed")
    hash_for = lambda process: hashlib.sha256((config["dev_process_hash_seed"] + "|" +
                                               process.lower()).encode("ascii")).hexdigest()
    eligible_dev = [process for process in ids if old[process]["v4_role"] == "fit"]
    dev = set(sorted(eligible_dev, key=hash_for)[:config["dev_process_count"]])
    paired, missing = Counter(), Counter()
    with gzip.open(ELIGIBILITY, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            activation = activations[row["activation"]]
            if activation["role"] != "train":
                continue
            process = activation["process_id"]
            if row["paired_eligible"] == "1":
                paired[process] += 1
            elif row["failure_reasons"] in MISSING:
                missing[process] += 1
    rows = [{"process_id": process, "v5_role": "dev" if process in dev else "fit",
             "paired_eligible_tiles": paired[process], "unqualified_s2_tiles": missing[process],
             "split_hash": hash_for(process)} for process in sorted(ids)]
    if len(dev) != 16 or any(row["paired_eligible_tiles"] < 5 for row in rows):
        raise RuntimeError("V5 process split incomplete")
    if OUT.exists() or META.exists():
        raise RuntimeError("Refusing to overwrite v5 split")
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fit_counts = [paired[p] for p in ids if p not in dev]
    budget_unique = sum(min(count, config["epochs"] *
                            config["training_chips_per_process_per_epoch"]) for count in fit_counts)
    META.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "hash-only 16 development IDs from prior v4 fit; no label pixels or scores read",
        "config_sha256": sha256(CONFIG), "original_split_sha256": sha256(SPLIT),
        "v4_split_sha256": sha256(V4_SPLIT), "eligibility_sha256": sha256(ELIGIBILITY),
        "output_sha256": sha256(OUT), "fit_processes": 65, "dev_processes": 16,
        "fit_paired_tiles": sum(fit_counts), "dev_paired_tiles": sum(paired[p] for p in dev),
        "planned_unique_fit_tiles": budget_unique,
        "dev_real_missing_processes_at_least_two": sum(missing[p] >= 2 for p in dev),
        "dev_real_missing_tiles": sum(missing[p] for p in dev),
        "test_labels_or_metrics_read": False}, indent=2) + "\n", encoding="utf-8")
    print(META.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
