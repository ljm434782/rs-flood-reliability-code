"""Index or download the immutable STURM-Flood Zenodo archive."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
RECORD_URL = "https://zenodo.org/api/records/12748983"
TARGET = ROOT / "data" / "raw" / "sturm" / "Dataset.zip"
MANIFEST = ROOT / "data" / "manifests" / "sturm_manifest.csv"
METADATA = ROOT / "data" / "manifests" / "sturm_zenodo_record.json"


def digest(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 - Zenodo publishes MD5 for integrity checks.
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def write_manifest(record: dict, asset: dict, *, state: str, sha256: str = "") -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "record_id", "doi", "version", "asset", "expected_bytes", "expected_checksum",
        "sha256", "state", "source_url", "downloaded_utc", "local_path", "license",
    ]
    row = {
        "record_id": record["id"],
        "doi": record["doi"],
        "version": record["metadata"].get("version", ""),
        "asset": asset["key"],
        "expected_bytes": asset["size"],
        "expected_checksum": asset["checksum"],
        "sha256": sha256,
        "state": state,
        "source_url": asset["links"]["self"],
        "downloaded_utc": datetime.now(timezone.utc).isoformat() if state == "verified" else "",
        "local_path": str(TARGET.relative_to(ROOT)) if state == "verified" else "",
        "license": record["metadata"]["license"]["id"],
    }
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def download(asset: dict) -> str:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    partial = TARGET.with_suffix(".zip.part")
    if TARGET.exists():
        if TARGET.stat().st_size != asset["size"]:
            raise RuntimeError("Existing archive size differs from Zenodo; inspect manually")
        existing_md5, existing_sha256 = digest(TARGET)
        if asset["checksum"] != f"md5:{existing_md5}":
            raise RuntimeError("Existing archive failed Zenodo MD5 check")
        return existing_sha256

    offset = partial.stat().st_size if partial.exists() else 0
    if offset > asset["size"]:
        raise RuntimeError("Partial archive is larger than expected")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(asset["links"]["self"], headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
            raise RuntimeError("Server did not honor resume range; partial file was kept unchanged")
        mode = "ab" if offset else "wb"
        last_report = offset
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    offset += len(chunk)
                    if offset - last_report >= 256 * 1024 * 1024:
                        print(f"Downloaded {offset / 1e9:.2f} / {asset['size'] / 1e9:.2f} GB", flush=True)
                        last_report = offset
    if partial.stat().st_size != asset["size"]:
        raise RuntimeError("Incomplete download; rerun with --download to resume")
    actual_md5, actual_sha256 = digest(partial)
    if asset["checksum"] != f"md5:{actual_md5}":
        raise RuntimeError("Downloaded archive failed Zenodo MD5 check")
    partial.replace(TARGET)
    return actual_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download and verify the 3.99 GB archive")
    args = parser.parse_args()
    response = requests.get(RECORD_URL, timeout=30)
    response.raise_for_status()
    record = response.json()
    assets = [item for item in record["files"] if item["key"] == "Dataset.zip"]
    if len(assets) != 1:
        raise RuntimeError("Expected exactly one STURM Dataset.zip asset")
    asset = assets[0]
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    METADATA.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Zenodo {record['doi']}: {asset['key']} ({asset['size']} bytes), {asset['checksum']}")
    if args.download:
        sha256 = download(asset)
        write_manifest(record, asset, state="verified", sha256=sha256)
        print(f"Verified {TARGET}; SHA256 {sha256}")
    else:
        write_manifest(record, asset, state="indexed")
        print("Indexed only; use --download to fetch the archive")


if __name__ == "__main__":
    main()
