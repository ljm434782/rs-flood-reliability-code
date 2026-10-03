"""Save official CEMS activation metadata for ImpactMesh process identity review."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://mapping.emergency.copernicus.eu/activations/api/activations/?limit=2000"
RAW = ROOT / "data" / "raw" / "cems" / "activations_api_20260927.json"
OUT = ROOT / "data" / "manifests" / "impactmesh_cems_activations.csv"
META = ROOT / "data" / "manifests" / "impactmesh_cems_activations_meta.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value: str) -> str:
    return " ".join(str(value or "").split())


def main() -> None:
    official = (ROOT / "data" / "manifests" / "impactmesh_flood_train.txt").read_text(encoding="utf-8").splitlines()
    activations = {sample_id.split("_", 1)[0] for sample_id in official}
    if len(activations) != 182:
        raise RuntimeError("Official ImpactMesh train activation count changed")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        response = requests.get(SOURCE, timeout=60)
        response.raise_for_status()
        RAW.write_bytes(response.content)
    payload = json.loads(RAW.read_text(encoding="utf-8"))
    if payload.get("next") is not None or len(payload["results"]) != payload["count"]:
        raise RuntimeError("CEMS API response was paginated or incomplete")
    by_code = {item["code"]: item for item in payload["results"]}
    missing = activations - set(by_code)
    if missing:
        raise RuntimeError(f"Missing official CEMS activation metadata: {sorted(missing)}")
    rows = []
    for code in sorted(activations):
        item = by_code[code]
        rows.append({
            "activation": code, "name": clean(item.get("name", "")),
            "countries": ";".join(clean(country["short_name"]) for country in item.get("countries", [])),
            "category": clean(item.get("category", {}).get("name", "")),
            "activation_time": item.get("activationTime", ""),
            "last_update": item.get("lastUpdate", ""),
            "centroid_wkt": item.get("centroid", ""),
            "activation_reason_excerpt": clean(item.get("search_snippet", "")),
            "source_url": f"https://mapping.emergency.copernicus.eu/activations/{code}/",
        })
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    META.write_text(json.dumps({
        "source_url": SOURCE, "indexed_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": RAW.relative_to(ROOT).as_posix(), "raw_sha256": sha256(RAW),
        "raw_count": payload["count"], "impactmesh_train_activation_count": len(rows),
        "output_sha256": sha256(OUT), "use": "process identity review only; no test labels or model tuning",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Indexed {len(rows)} official CEMS activation records", flush=True)


if __name__ == "__main__":
    main()
