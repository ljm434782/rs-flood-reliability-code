"""Create a label-blind candidate activation split before full quality scanning.

The 24 already inspected official-train activations remain in development train.
This is a candidate manifest, not a frozen preregistration or sample admission.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from flood_reliability.splits import require_process_disjoint


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
OFFICIAL_TRAIN = MANIFESTS / "impactmesh_flood_train.txt"
OFFICIAL_HOLDOUT = MANIFESTS / "impactmesh_flood_test_holdout.txt"
FRAME = MANIFESTS / "impactmesh_temporal_audit_frame.csv"
OUTPUT = MANIFESTS / "impactmesh_process_split_candidate.csv"
META = MANIFESTS / "impactmesh_process_split_candidate_meta.json"
SEED = "impactmesh-v1-process-split-20260926"
N_INTERNAL_TEST = 20
N_CALIBRATION = 20
N_VALIDATION = 20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def activations(path: Path) -> Counter:
    ids = [value for value in path.read_text(encoding="utf-8").splitlines() if value]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate sample IDs: {path}")
    return Counter(sample_id.split("_", 1)[0] for sample_id in ids)


def main() -> None:
    train_counts = activations(OFFICIAL_TRAIN)
    holdout_counts = activations(OFFICIAL_HOLDOUT)
    if len(train_counts) != 182 or len(holdout_counts) != 12 or set(train_counts) & set(holdout_counts):
        raise RuntimeError("Official activation counts or train/holdout separation changed")
    with FRAME.open(newline="", encoding="utf-8") as handle:
        audited = {row["activation"] for row in csv.DictReader(handle)}
    if len(audited) != 24 or not audited <= set(train_counts):
        raise RuntimeError("Fixed audit activations no longer match official train")
    untouched = sorted(set(train_counts) - audited,
                       key=lambda activation: (hashlib.sha256(f"{SEED}:{activation}".encode()).hexdigest(), activation))
    roles = {}
    for activation in audited:
        roles[activation] = "train"
    cuts = (N_INTERNAL_TEST, N_INTERNAL_TEST + N_CALIBRATION,
            N_INTERNAL_TEST + N_CALIBRATION + N_VALIDATION)
    for position, activation in enumerate(untouched):
        roles[activation] = ("internal_test" if position < cuts[0]
                             else "calibration" if position < cuts[1]
                             else "validation" if position < cuts[2]
                             else "train")
    for activation in holdout_counts:
        roles[activation] = "official_test_holdout"
    groups = {role: sorted(activation for activation, assigned in roles.items() if assigned == role)
              for role in ("train", "validation", "calibration", "internal_test", "official_test_holdout")}
    require_process_disjoint(groups)
    rows = [{"activation": activation, "role": role,
             "official_source": "test_holdout" if role == "official_test_holdout" else "train",
             "official_source_tiles": holdout_counts[activation] if role == "official_test_holdout" else train_counts[activation],
             "already_inspected_train_pilot": int(activation in audited)}
            for activation, role in sorted(roles.items())]
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    META.write_text(json.dumps({
        "state": "candidate process split only; no sample admission or preregistration freeze",
        "seed": SEED,
        "rule": "24 already inspected train activations locked to train; remaining official-train activations SHA256 sorted, first 20 internal test, next 20 calibration, next 20 validation, rest train; 12 official test_holdout activations locked final",
        "official_train_sha256": sha256(OFFICIAL_TRAIN),
        "official_test_holdout_sha256": sha256(OFFICIAL_HOLDOUT),
        "fixed_audit_frame_sha256": sha256(FRAME),
        "counts": {role: len(ids) for role, ids in groups.items()},
        "use_limit": "No eligibility threshold may be chosen using internal_test or official_test_holdout; process identity still needs CEMS date/space review",
        "output_sha256": sha256(OUTPUT),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({role: len(ids) for role, ids in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
