"""Freeze a label-blind train-only pilot sample for ImpactMesh timing/QC audit."""

import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
SOURCE = MANIFESTS / "impactmesh_train_mask_index.csv"
FRAME = MANIFESTS / "impactmesh_temporal_audit_frame.csv"
META = MANIFESTS / "impactmesh_temporal_audit_frame_meta.json"
SEED = "impactmesh-temporal-pilot-v1-20260926"
PROCESS_COUNT = 24


def digest(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).hexdigest()


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_activation = defaultdict(list)
    for row in rows:
        if row["activation"] != row["sample_id"].split("_", 1)[0]:
            raise RuntimeError(f"Activation mismatch: {row['sample_id']}")
        by_activation[row["activation"]].append(row)
    official = set((MANIFESTS / "impactmesh_flood_train.txt").read_text(encoding="utf-8").splitlines())
    if len(rows) != len(official) or {row["sample_id"] for row in rows} != official:
        raise RuntimeError("Train mask index differs from official train sample list")
    if len(by_activation) != 182 or PROCESS_COUNT > len(by_activation):
        raise RuntimeError("Unexpected activation count")

    selected_activations = sorted(by_activation, key=lambda activation: (digest(f"activation|{activation}"), activation))[:PROCESS_COUNT]
    selected = []
    for rank, activation in enumerate(selected_activations, 1):
        choice = min(by_activation[activation], key=lambda row: (digest(f"sample|{row['sample_id']}"), row["sample_id"]))
        selected.append({
            "rank": rank,
            "activation": activation,
            "sample_id": choice["sample_id"],
            "train_mask_tar_ordinal": choice["ordinal"],
            "train_activation_sample_count": len(by_activation[activation]),
            "selection_state": "fixed before S1/S2 pilot retrieval; label pixels unread",
        })
    metadata = {
        "purpose": "train-only feasibility pilot for S1/S2 timing, SCL/cloud/nodata, and label-source follow-up",
        "seed": SEED,
        "source_index": SOURCE.name,
        "source_index_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_mask_tar_lfs_sha256": json.loads((MANIFESTS / "impactmesh_train_mask_index_meta.json").read_text(encoding="utf-8"))["lfs_sha256"],
        "source_train_split_sha256": hashlib.sha256((MANIFESTS / "impactmesh_flood_train.txt").read_bytes()).hexdigest(),
        "eligible_train_activations": len(by_activation),
        "selected_activations": PROCESS_COUNT,
        "selection_rule": "sort all train activations by SHA256(seed|activation|ID), take first 24; within each, choose minimum SHA256(seed|sample|ID)",
        "selection_inputs": "official train sample IDs and TAR member names/ordinals only; no labels, imagery, clouds, timestamps, validation or test results enter the hash ranking",
        "prior_audit_context": "four convenience triplets from train/val/test/test_holdout had been inspected before this frame was generated; none are selected by their observed quality",
        "use_limit": "pilot, not final training split; report all 24 activations including unusable ones; no model selection or test-threshold tuning",
    }
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(selected[0]))
    writer.writeheader()
    writer.writerows(selected)
    frame_text = buffer.getvalue()
    meta_text = json.dumps(metadata, indent=2)
    if FRAME.exists() and FRAME.read_bytes() != frame_text.encode("utf-8"):
        raise RuntimeError("Frozen pilot frame differs from current deterministic selection")
    if META.exists():
        previous = json.loads(META.read_text(encoding="utf-8"))
        if any(previous.get(key) != metadata[key] for key in ("seed", "source_index_sha256", "eligible_train_activations", "selected_activations")):
            raise RuntimeError("Frozen pilot metadata differs from current inputs")
    FRAME.write_text(frame_text, encoding="utf-8", newline="")
    META.write_text(meta_text, encoding="utf-8")
    print(f"Fixed {len(selected)} train samples from {len(by_activation)} activations")
    print(f"TAR ordinal range: {min(int(row['train_mask_tar_ordinal']) for row in selected)}–{max(int(row['train_mask_tar_ordinal']) for row in selected)}")
    print(FRAME)


if __name__ == "__main__":
    main()
