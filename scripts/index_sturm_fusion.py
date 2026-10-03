"""Read public STURM-Fusion-24 ZIP metadata without downloading the full archive."""

import csv
import json
import zipfile
from pathlib import Path

import requests

from index_sturm_zip import RangeReader


ROOT = Path(__file__).resolve().parents[1]
REPO = "taylor-millin-reade/STURM-fusion-24"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/Dataset.zip"


def main() -> None:
    response = requests.get(API, timeout=60)
    response.raise_for_status()
    asset = next(item for item in response.json() if item.get("path") == "Dataset.zip")
    manifest = ROOT / "data" / "manifests" / "sturm_fusion_archive.json"
    manifest.write_text(json.dumps({"repository": REPO, "asset": asset, "source_url": URL}, indent=2), encoding="utf-8")
    with zipfile.ZipFile(RangeReader(URL, asset["size"])) as archive:
        infos = archive.infolist()
        destination = ROOT / "data" / "raw" / "sturm_fusion" / "metadata.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read("Dataset/metadata/metadata.csv"))
    with (ROOT / "data" / "manifests" / "sturm_fusion_zip_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "size_bytes", "compressed_bytes", "crc32"])
        for item in infos:
            writer.writerow([item.filename, item.file_size, item.compress_size, f"{item.CRC:08x}"])
    print(f"Indexed {len(infos)} ZIP members, extracted metadata; full archive not downloaded")
    print(f"Archive SHA256 from Hugging Face LFS: {asset['lfs']['oid']}")


if __name__ == "__main__":
    main()
