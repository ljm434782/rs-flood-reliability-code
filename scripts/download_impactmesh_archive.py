"""Resume and verify one immutable official ImpactMesh v1 image archive."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data" / "manifests" / "impactmesh_assets.json"
RAW = ROOT / "data" / "raw" / "impactmesh"
MANIFEST = ROOT / "data" / "manifests" / "impactmesh_archive_downloads.json"
REPO = "ibm-esa-geospatial/ImpactMesh-Flood"
CHUNK = 8 * 1024 * 1024
DEFAULT_ENDPOINT = "https://huggingface.co"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(modality: str, endpoint: str, *, direct: bool, split: str = "train") -> None:
    name = f"{split}/{modality}.tar"
    assets = {item["path"]: item for item in json.loads(ASSETS.read_text(encoding="utf-8"))["assets"]}
    asset = assets[name]
    expected_size = int(asset["size"])
    expected_hash = asset["lfs"]["oid"]
    endpoint = endpoint.rstrip("/")
    url = f"{endpoint}/datasets/{REPO}/resolve/main/{name}"
    out = RAW / split
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{modality}.tar"
    partial = out / f"{modality}.tar.part"
    if target.exists():
        if target.stat().st_size != expected_size or sha256(target) != expected_hash:
            raise RuntimeError(f"Existing raw archive differs from official SHA256: {target}")
        print(f"Already verified: {target}", flush=True)
        return
    if partial.exists() and partial.stat().st_size > expected_size:
        raise RuntimeError(f"Partial archive exceeds official size: {partial}")
    offset = partial.stat().st_size if partial.exists() else 0
    last_report = offset
    session = requests.Session()
    session.trust_env = not direct
    failures = 0
    while offset < expected_size:
        try:
            with session.get(url, headers={"Range": f"bytes={offset}-{expected_size - 1}"},
                             stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                wanted = f"bytes {offset}-{expected_size - 1}/{expected_size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != wanted:
                    raise RuntimeError(f"Server did not honor exact resume range: {response.status_code} {response.headers.get('Content-Range')}")
                with partial.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        offset += len(chunk)
                        if offset > expected_size:
                            raise RuntimeError("Archive exceeds official size")
                        if offset - last_report >= 512 * 1024 * 1024:
                            print(f"{modality}: {offset / 1e9:.2f} / {expected_size / 1e9:.2f} GB", flush=True)
                            last_report = offset
            failures = 0
        except (requests.RequestException, OSError) as error:
            failures += 1
            if failures > 6:
                raise RuntimeError(f"Download interrupted at byte {offset}; partial retained") from error
            offset = partial.stat().st_size if partial.exists() else 0
            print(f"Retry {failures} at byte {offset}: {error}", flush=True)
            time.sleep(min(30, 2 ** failures))
    if partial.stat().st_size != expected_size:
        raise RuntimeError("Incomplete archive; partial retained")
    digest = sha256(partial)
    if digest != expected_hash:
        raise RuntimeError(f"Official SHA256 mismatch; partial retained: {digest}")
    partial.replace(target)
    record = {
        "repository": REPO,
        "version": "v1",
        "license": "CC-BY-4.0",
        "license_source": f"https://huggingface.co/datasets/{REPO}",
        "path": name,
        "source_url": url,
        "download_endpoint": endpoint,
        "proxy_environment_used": not direct,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "size_bytes": expected_size,
        "sha256": digest,
        "local_path": target.relative_to(ROOT).as_posix(),
        "state": "full-official-archive-verified",
    }
    manifest_path = MANIFEST.with_name(f"impactmesh_archive_download_{split}_{modality}.json")
    temp = manifest_path.with_suffix(".json.part")
    temp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temp.replace(manifest_path)
    print(f"Verified {target}: SHA256 {digest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("S1RTC", "S2L2A"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                        help="Hugging Face-compatible endpoint")
    parser.add_argument("--direct", action="store_true",
                        help="Ignore HTTP(S)_PROXY and other environment proxy settings")
    args = parser.parse_args()
    download(args.modality, args.endpoint, direct=args.direct, split=args.split)


if __name__ == "__main__":
    main()
