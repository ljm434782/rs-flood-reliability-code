"""Index public ImpactMesh-Flood assets and official split identifiers only."""

import json
from collections import Counter
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
REPO = "ibm-esa-geospatial/ImpactMesh-Flood"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"
OUT = ROOT / "data" / "manifests"


def main() -> None:
    response = requests.get(API, timeout=60)
    response.raise_for_status()
    assets = [item for item in response.json() if item.get("type") == "file"]
    (OUT / "impactmesh_assets.json").write_text(json.dumps({"repository": REPO, "assets": assets}, indent=2), encoding="utf-8")
    groups = {}
    for name in ("train", "val", "test", "test_holdout", "test_seen_events"):
        filename = f"impactmesh_flood_{name}.txt"
        response = requests.get(f"{BASE}/split/{filename}", timeout=60)
        response.raise_for_status()
        (OUT / filename).write_bytes(response.content)
        prefixes = [line.strip() for line in response.text.splitlines() if line.strip()]
        groups[name] = Counter(prefix.split("_", 1)[0] for prefix in prefixes)
        print(f"{name}: {len(prefixes)} samples, {len(groups[name])} activation IDs")
    print("Train/test activation overlap:", len(set(groups["train"]) & set(groups["test"])))
    print("Train/test_holdout activation overlap:", len(set(groups["train"]) & set(groups["test_holdout"])))
    print("Total distinct activation IDs:", len(set().union(*(set(group) for group in groups.values()))))


if __name__ == "__main__":
    main()
