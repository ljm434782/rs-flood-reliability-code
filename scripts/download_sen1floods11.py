"""Index Sen1Floods11 v1.1 QC objects and optionally download its held-out test split."""

import argparse
import base64
import csv
import hashlib
from pathlib import Path
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[1]
BUCKET = "sen1floods11"
BASE = "https://storage.googleapis.com"
API = f"{BASE}/storage/v1/b/{BUCKET}/o"
RAW = ROOT / "data" / "raw" / "sen1floods11"
MANIFEST = ROOT / "data" / "manifests" / "sen1floods11_manifest.csv"
TEST_SPLIT = "v1.1/splits/flood_handlabeled/flood_test_data.csv"
METADATA = "v1.1/Sen1Floods11_Metadata.geojson"
PREFIXES = (
    "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "v1.1/data/flood_events/HandLabeled/LabelHand/",
)


def list_objects(session: requests.Session, prefix: str) -> list[dict]:
    objects = []
    page_token = None
    while True:
        params = {"prefix": prefix, "maxResults": 1000}
        if page_token:
            params["pageToken"] = page_token
        response = session.get(API, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        objects.extend(payload.get("items", []))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return objects


def get_object(session: requests.Session, name: str) -> dict:
    response = session.get(API + "/" + quote(name, safe=""), timeout=30)
    response.raise_for_status()
    return response.json()


def download_object(session: requests.Session, item: dict) -> tuple[str, str]:
    name = item["name"]
    target = RAW / name.removeprefix("v1.1/")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        response = session.get(f"{BASE}/{BUCKET}/{name}", timeout=(30, 120))
        response.raise_for_status()
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(response.content)
        temporary.replace(target)
    content = target.read_bytes()
    if len(content) != int(item["size"]):
        raise RuntimeError(f"Size mismatch: {name}")
    expected_md5 = base64.b64decode(item["md5Hash"])
    if hashlib.md5(content).digest() != expected_md5:  # noqa: S324 - GCS publishes MD5.
        raise RuntimeError(f"GCS MD5 mismatch: {name}")
    return str(target.relative_to(ROOT)), hashlib.sha256(content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-test", action="store_true", help="Fetch the official hand-labeled QC test S1 and labels")
    args = parser.parse_args()
    with requests.Session() as session:
        objects = {item["name"]: item for prefix in PREFIXES for item in list_objects(session, prefix)}
        objects[TEST_SPLIT] = get_object(session, TEST_SPLIT)
        objects[METADATA] = get_object(session, METADATA)
        selected = {TEST_SPLIT, METADATA}
        split_path, _ = download_object(session, objects[TEST_SPLIT])
        download_object(session, objects[METADATA])
        with (ROOT / split_path).open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                if len(row) != 2:
                    raise RuntimeError(f"Unexpected hand-labeled split row: {row}")
                image, label = row
                selected.add(PREFIXES[0] + image)
                selected.add(PREFIXES[1] + label)
        missing = selected - objects.keys()
        if missing:
            raise RuntimeError(f"Official test references missing bucket objects: {sorted(missing)[:3]}")
        rows = []
        for name, item in sorted(objects.items()):
            local_path = ""
            sha256 = ""
            state = "indexed"
            if name in (TEST_SPLIT, METADATA) or (args.download_test and name in selected):
                local_path, sha256 = download_object(session, item)
                state = "verified"
            rows.append({
                "name": name, "size_bytes": item["size"], "gcs_md5_base64": item.get("md5Hash", ""),
                "sha256": sha256, "state": state, "selected_test": name in selected,
                "local_path": local_path, "source_url": f"{BASE}/{BUCKET}/{name}",
            })
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Indexed {len(objects)} GCS objects; official test split selects {len(selected)-2} images/labels")
    print(f"Verified {sum(row['state'] == 'verified' for row in rows)} objects; manifest {MANIFEST}")


if __name__ == "__main__":
    main()
