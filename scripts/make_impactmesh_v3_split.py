"""Freeze process-disjoint v3 fit/development roles from training input metadata."""

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.train_impactmesh_formal import ELIGIBILITY, ROOT, SPLIT, sha256


CONFIG = ROOT / "configs" / "impactmesh_exploratory_v3.json"
OUT = ROOT / "data" / "manifests" / "impactmesh_v3_training_dev_processes.csv"
META = OUT.with_suffix(".json")
REASONS = {"visible_below_0.5", "no_pre_s1_s2_within_30d"}


def main():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with SPLIT.open(newline="", encoding="utf-8") as handle:
        activations = {row["activation"]: row for row in csv.DictReader(handle)}
    ids = sorted({row["process_id"] for row in activations.values() if row["role"] == "train"},
                 key=lambda process: hashlib.sha256((
                     config["dev_process_hash_seed"] + "|" + process.lower()).encode("ascii")).hexdigest())
    if len(ids) != 81:
        raise RuntimeError("Original training process count changed")
    dev = set(ids[:config["dev_process_count"]])
    paired = Counter()
    missing = Counter()
    with gzip.open(ELIGIBILITY, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group = activations[row["activation"]]
            if group["role"] != "train":
                continue
            process = group["process_id"]
            if row["paired_eligible"] == "1":
                paired[process] += 1
            elif row["failure_reasons"] in REASONS:
                missing[process] += 1
    rows = []
    for process in sorted(ids):
        if paired[process] < 5:
            raise RuntimeError(f"Original paired process below frozen minimum: {process}")
        rows.append({"process_id": process, "v3_role": "dev" if process in dev else "fit",
                     "paired_eligible_tiles": paired[process],
                     "unqualified_s2_tiles": missing[process],
                     "split_hash": hashlib.sha256((config["dev_process_hash_seed"] + "|" +
                                                   process.lower()).encode("ascii")).hexdigest()})
    if OUT.exists() or META.exists():
        raise RuntimeError("Refusing to overwrite v3 split")
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "method": "hash-only split of original training process IDs; no label pixels read",
            "config_sha256": sha256(CONFIG), "original_split_sha256": sha256(SPLIT),
            "eligibility_sha256": sha256(ELIGIBILITY), "output_sha256": sha256(OUT),
            "fit_processes": len(ids) - len(dev), "dev_processes": len(dev),
            "fit_paired_tiles": sum(paired[p] for p in ids if p not in dev),
            "dev_paired_tiles": sum(paired[p] for p in dev),
            "dev_real_missing_processes": sum(missing[p] >= 2 for p in dev),
            "dev_real_missing_tiles": sum(missing[p] for p in dev),
            "test_labels_or_metrics_read": False}
    META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
