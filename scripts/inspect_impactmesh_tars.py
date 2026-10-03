"""Inspect the first few members of ImpactMesh validation tars via HTTP ranges."""

import json
import tarfile
from pathlib import Path

from index_sturm_zip import RangeReader


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data" / "manifests" / "impactmesh_assets.json"
REPO = "ibm-esa-geospatial/ImpactMesh-Flood"


def main() -> None:
    assets = {item["path"]: item for item in json.loads(ASSETS.read_text(encoding="utf-8"))["assets"]}
    for modality in ("MASK", "S1RTC", "S2L2A"):
        path = f"val/{modality}.tar"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
        with tarfile.open(fileobj=RangeReader(url, assets[path]["size"], block_size=1024 * 1024), mode="r:") as archive:
            members = []
            for _ in range(6):
                member = archive.next()
                if member is None:
                    break
                members.append((member.name, member.size, member.offset_data))
        print(modality, members, flush=True)


if __name__ == "__main__":
    main()
