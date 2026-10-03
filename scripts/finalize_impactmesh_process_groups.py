"""Conservatively group plausible shared flood episodes before formal fitting.

Only official-train acquisition times, TIFF geometry, and public CEMS event
descriptions inform this amendment. No label pixels or model scores are read.
The proximity screen is a conservative blocking device, not a claim that every
edge is the same hydrological flood.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from flood_reliability.splits import require_process_disjoint


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
PAIRS = MANIFESTS / "impactmesh_process_neighbor_candidates.csv"
COUNTS = MANIFESTS / "impactmesh_all_train_frozen_process_counts.csv"
FRAME = MANIFESTS / "impactmesh_temporal_audit_frame.csv"
OLD_SPLIT = MANIFESTS / "impactmesh_process_split_frozen.csv"
OUTPUT = MANIFESTS / "impactmesh_process_split_grouped.csv"
GROUPS = MANIFESTS / "impactmesh_process_groups.csv"
META = MANIFESTS / "impactmesh_process_split_grouped_meta.json"
SEED = "impactmesh-v1-final-process-split-20260927"

# Four pairs have independent public-event evidence of a shared episode.
# The other proximity candidates remain distinct unless their event identity
# is established; proximity alone cannot prove a common flood process.
MERGE = {
    frozenset(("EMSR228", "EMSR231")),  # NW Ireland cross-border August 2017
    frozenset(("EMSR421", "EMSR422")),  # Storm Gloria France / Spain
    frozenset(("EMSR777", "EMSR778")),  # Tropical Storm Sara
    frozenset(("EMSR550", "EMSR554")),  # Australia Goondiwindi / Wee Waa
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with COUNTS.open(newline="", encoding="utf-8") as handle:
        counts = {row["activation"]: row for row in csv.DictReader(handle)}
    with OLD_SPLIT.open(newline="", encoding="utf-8") as handle:
        old = {row["activation"]: row for row in csv.DictReader(handle)}
    with FRAME.open(newline="", encoding="utf-8") as handle:
        pilots = {row["activation"] for row in csv.DictReader(handle)}
    if len(counts) != 182 or len(pilots) != 24 or len(old) != 194:
        raise RuntimeError("Expected official train counts, fixed pilots, and prior split")
    parent = {activation: activation for activation in counts}

    def find(activation: str) -> str:
        if parent[activation] != activation:
            parent[activation] = find(parent[activation])
        return parent[activation]

    screened = 0
    grouped = 0
    merged_pairs = []
    with PAIRS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            a, b = row["activation_a"], row["activation_b"]
            if a not in counts or b not in counts:
                raise RuntimeError("Proximity edge outside official train")
            screened += 1
            if frozenset((a, b)) not in MERGE:
                continue
            parent[find(b)] = find(a)
            grouped += 1
            merged_pairs.append([a, b])
    if {frozenset(pair) for pair in merged_pairs} != MERGE:
        raise RuntimeError("Documented shared-event edges changed")
    members = defaultdict(list)
    for activation in counts:
        members[find(activation)].append(activation)
    components = [sorted(values) for values in members.values()]
    canonical = {value: component[0] for component in components for value in component}
    eligible = []
    for component in components:
        total = sum(int(counts[value]["paired_eligible_samples"]) for value in component)
        if total >= 5:
            eligible.append(component)
    pilot_components = {canonical[activation] for activation in pilots}
    nonpilot = sorted(
        (component for component in eligible if component[0] not in pilot_components),
        key=lambda component: (hashlib.sha256(
            f"{SEED}|{component[0].lower()}".encode("ascii")).hexdigest(), component[0]),
    )
    if len(nonpilot) < 60:
        raise RuntimeError("Frozen minimum of 60 eligible non-pilot process groups not met")
    group_role = {component[0]: "train" for component in eligible if component[0] in pilot_components}
    for index, component in enumerate(nonpilot):
        group_role[component[0]] = (
            "internal_test" if index < 20 else
            "calibration" if index < 40 else
            "validation" if index < 60 else "train")
    group_rows = []
    for component in sorted(components):
        key = component[0]
        group_rows.append({
            "process_id": key,
            "activations": ";".join(component),
            "activation_count": len(component),
            "paired_eligible_samples": sum(int(counts[a]["paired_eligible_samples"]) for a in component),
            "pilot_locked_train": int(key in pilot_components),
            "role": group_role.get(key, "excluded_paired"),
        })
    write_csv(GROUPS, group_rows)
    split_rows = []
    for activation in sorted(counts):
        key = canonical[activation]
        split_rows.append({
            **counts[activation], "process_id": key,
            "role": group_role.get(key, "excluded_paired"),
            "split_hash": hashlib.sha256(
                f"{SEED}|{key.lower()}".encode("ascii")).hexdigest(),
            "source": "official_train",
        })
    for activation in sorted(set(old) - set(counts)):
        split_rows.append({
            "activation": activation, "official_train_samples": 0,
            "paired_eligible_samples": "", "paired_process_eligible": "",
            "fixed_pilot": 0, "process_id": activation,
            "role": "official_test_holdout", "split_hash": "",
            "source": "official_test_holdout",
        })
    write_csv(OUTPUT, split_rows)
    roles = ("train", "validation", "calibration", "internal_test", "official_test_holdout")
    require_process_disjoint({role: [r["process_id"] for r in split_rows if r["role"] == role]
                              for role in roles})
    if any(group_role.get(canonical[a]) != "train" for a in pilots if canonical[a] in group_role):
        raise RuntimeError("Inspected pilot escaped train")
    result = {
        "purpose": "pretraining process-identity amendment; no label pixels or model scores",
        "screen": "official-train S1 date ranges within 30 days and activation footprint centroids within 500 km",
        "interpretation": "Four event-supported merges; proximity alone does not prove a shared flood process",
        "merged_event_pairs": sorted(merged_pairs),
        "screened_edges": screened, "grouped_edges": grouped,
        "groups": len(components), "eligible_groups": len(eligible),
        "eligible_nonpilot_groups": len(nonpilot),
        "role_group_counts": dict(Counter(row["role"] for row in group_rows)),
        "role_activation_counts": dict(Counter(row["role"] for row in split_rows)),
        "seed": SEED,
        "input_sha256": {p.name: digest(p) for p in (PAIRS, COUNTS, FRAME, OLD_SPLIT)},
        "output_sha256": {p.name: digest(p) for p in (OUTPUT, GROUPS)},
    }
    META.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
