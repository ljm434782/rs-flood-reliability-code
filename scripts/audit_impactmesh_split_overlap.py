"""Audit official ImpactMesh split process overlap without modifying splits."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flood_reliability.splits import process_overlap_report


MANIFESTS = ROOT / "data" / "manifests"
OUTPUT = MANIFESTS / "impactmesh_split_overlap_audit.json"


def read_ids(name: str) -> list[str]:
    return (MANIFESTS / f"impactmesh_flood_{name}.txt").read_text(encoding="utf-8").splitlines()


def main() -> None:
    samples = {name: read_ids(name) for name in ("train", "val", "test")}
    holdout = set(read_ids("test_holdout"))
    if not holdout <= set(samples["test"]):
        raise RuntimeError("Official test_holdout is not a subset of test")
    if sum(len(ids) for ids in samples.values()) != len(set().union(*(set(ids) for ids in samples.values()))):
        raise RuntimeError("Official train/val/test sample IDs are duplicated")
    # CEMS activation is a conservative proxy; final flood-process IDs require auditing.
    activations = {name: {sample.split("_", 1)[0] for sample in ids} for name, ids in samples.items()}
    overlaps = process_overlap_report(activations)
    holdout_activations = {sample.split("_", 1)[0] for sample in holdout}
    result = {
        "process_id_rule": "sample ID prefix before first underscore = CEMS activation; final process grouping unverified",
        "sample_counts": {name: len(ids) for name, ids in samples.items()},
        "activation_counts": {name: len(ids) for name, ids in activations.items()},
        "pairwise_activation_overlaps": overlaps,
        "test_holdout_sample_count": len(holdout),
        "test_holdout_activation_count": len(holdout_activations),
        "test_holdout_activation_overlap_with_train_or_val": sorted(holdout_activations & (activations["train"] | activations["val"])),
        "state": "official release audit only; not an approved process-disjoint experimental split",
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print({pair: len(ids) for pair, ids in overlaps.items()})
    print(f"Holdout activation overlap with train/val: {len(result['test_holdout_activation_overlap_with_train_or_val'])}")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
