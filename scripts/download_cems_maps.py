"""Fetch selected official CEMS maps for label provenance audits."""

import csv
import hashlib
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://cems-mapping-website.s3.eu-west-1.amazonaws.com/static/activations"
MAPS = (
    ("EMSR470", "AOI01_DEL_MONIT01_r1_RTP01_v1", "EMSR470_AOI01_DEL_MONIT01_r1_RTP01_v1.pdf"),
    ("EMSR470", "AOI01_DEL_PRODUCT_r1_RTP01_v2", "EMSR470_AOI01_DEL_PRODUCT_r1_RTP01_v2.pdf"),
    ("EMSR264", "16MIANDRIVAZO_01DELINEATION_MAP_v2", "EMSR264_16MIANDRIVAZO_01DELINEATION_MAP_v2_200dpi.pdf"),
    ("EMSR251", "03KRAGERO_01DELINEATION_MAP_v2", "EMSR251_03KRAGERO_01DELINEATION_MAP_v2_200dpi.pdf"),
    ("EMSR251", "03KRAGERO_01DELINEATION_MONIT01_v1", "EMSR251_03KRAGERO_01DELINEATION_MONIT01_v1_200dpi.pdf"),
    ("EMSR251", "03KRAGERO_01DELINEATION_MONIT01_v1_vector", "EMSR251_03KRAGERO_01DELINEATION_MONIT01_v1_vector.zip"),
    ("EMSR424", "AOI01_FEP_PRODUCT_r1_RTP01_v2", "EMSR424_AOI01_FEP_PRODUCT_r1_RTP01_v2.pdf"),
    ("EMSR424", "AOI01_FEP_PRODUCT_r1_VECTORS_v1_vector", "EMSR424_AOI01_FEP_PRODUCT_r1_VECTORS_v1_vector.zip"),
    ("EMSR424", "AOI01_DEL_PRODUCT_r1_RTP01_v2", "EMSR424_AOI01_DEL_PRODUCT_r1_RTP01_v2.pdf"),
    ("EMSR424", "AOI01_DEL_PRODUCT_r1_RTP01_v2_vector", "EMSR424_AOI01_DEL_PRODUCT_r1_RTP01_v2_vector.zip"),
    ("EMSR424", "AOI01_DEL_MONIT01_r1_RTP01_v1", "EMSR424_AOI01_DEL_MONIT01_r1_RTP01_v1.pdf"),
    ("EMSR424", "AOI01_DEL_MONIT01_r1_RTP01_v1_vector", "EMSR424_AOI01_DEL_MONIT01_r1_RTP01_v1_vector.zip"),
)
INDEX = ROOT / "data" / "manifests" / "cems_index.csv"


def main() -> None:
    with INDEX.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for activation, product, filename in MAPS:
        folder = ROOT / "data" / "raw" / "cems" / activation
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / filename
        if not target.exists():
            response = requests.get(f"{BASE}/{activation}/{filename}", stream=True, timeout=60)
            response.raise_for_status()
            partial = target.with_suffix(target.suffix + ".part")
            with partial.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
            partial.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        relative = target.relative_to(ROOT).as_posix()
        existing = [row for row in rows if row["local_path"] == relative]
        if len(existing) > 1:
            raise RuntimeError(f"Duplicate CEMS manifest path: {relative}")
        if existing and existing[0]["sha256"] != digest:
            raise RuntimeError(f"Existing CEMS file differs from recorded SHA256: {relative}")
        if not existing:
            rows.append({
                "activation": activation,
                "product": product,
                "source_url": f"{BASE}/{activation}/{filename}",
                "local_path": relative,
                "sha256": digest,
                "source_image_time_utc": "",
                "status": "downloaded-unreviewed",
            })
        print(f"{relative}: {target.stat().st_size} bytes; SHA256 {digest}", flush=True)
    with INDEX.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
