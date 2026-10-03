"""Download only official CEMS delineation maps needed for source-time audit."""

import csv
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

from scripts.train_impactmesh_formal import ROOT, sha256

ROOT_RAW = ROOT / "data/raw/cems"
MAN = ROOT / "data/manifests"
OUT = MAN / "impactmesh_new_blind_legacy_cems_map_sources.csv"
META = OUT.with_suffix(".json")
AOIS = {"EMSR410": "01", "EMSR411": "01", "EMSR496": "01", "EMSR559": "04"}


def main():
    if OUT.exists() or META.exists():
        raise RuntimeError("Refusing to overwrite legacy CEMS source manifest")
    records = []
    for code, aoi in AOIS.items():
        page_url = f"https://mapping.emergency.copernicus.eu/activations/{code}/"
        page_file = ROOT_RAW / f"{code}_page_20260927.html"
        if page_file.exists():
            body = page_file.read_bytes()
        else:
            response = requests.get(page_url, timeout=30)
            response.raise_for_status()
            body = response.content
            page_file.write_bytes(body)
        urls = [html.unescape(value) for value in re.findall(
            r'href="([^"]+\.pdf[^"]*)"', body.decode("utf-8"))]
        selected = sorted(set(url for url in urls if f"{code}_AOI{aoi}_DEL_" in url))
        if not selected:
            raise RuntimeError(f"No official delineation map URLs for {code} AOI{aoi}")
        for url in selected:
            filename = url.rsplit("/", 1)[-1]
            path = ROOT_RAW / code / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                path.write_bytes(response.content)
            records.append({"activation": code, "aoi": aoi, "source_page": page_url,
                            "source_page_sha256": sha256(page_file), "map_url": url,
                            "local_path": path.relative_to(ROOT).as_posix(),
                            "bytes": path.stat().st_size, "sha256": sha256(path),
                            "downloaded_or_verified_utc": datetime.now(timezone.utc).isoformat(),
                            "purpose": "reference imagery time only; no ImpactMesh labels inspected"})
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    META.write_text(json.dumps({"completed_utc": datetime.now(timezone.utc).isoformat(),
        "source": "CEMS official activation pages and linked AOI delineation maps",
        "map_count": len(records), "output_sha256": sha256(OUT),
        "impactmesh_label_pixels_read": False, "model_outputs_read": False},
        indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"maps": len(records), "by_event": {code: sum(r["activation"] == code for r in records)
                                                     for code in AOIS}}, indent=2))


if __name__ == "__main__":
    main()
