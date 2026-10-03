"""Fetch the exact MIT-licensed SMAGNet source revision for a frozen run."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from flood_reliability.smagnet_adapter import AUTHOR_COMMIT, AUTHOR_SOURCE_SHA256


REPOSITORY = "https://github.com/ASUcicilab/SMAGNet.git"
LICENSE_SHA256 = "67b44fefa9e1d4ef9d71ea00097dbdc4e556d2ee0eec755f496051e3b9a75855"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    with tempfile.TemporaryDirectory(prefix="smagnet-fetch-") as temp:
        checkout = Path(temp) / "source"
        subprocess.run(["git", "clone", "--no-checkout", REPOSITORY, str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", AUTHOR_COMMIT], check=True)
        source = checkout / "src" / "smagnet.py"
        license_file = checkout / "LICENSE"
        if digest(source) != AUTHOR_SOURCE_SHA256:
            raise RuntimeError("Fetched smagnet.py does not match the frozen SHA-256")
        if digest(license_file) != LICENSE_SHA256:
            raise RuntimeError("Fetched LICENSE does not match the frozen SHA-256")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir()
        shutil.copy2(source, destination / "smagnet.py")
        shutil.copy2(license_file, destination / "LICENSE")
        (destination / "SOURCE.json").write_text(
            "{\n"
            f'  "repository": "{REPOSITORY}",\n'
            f'  "commit": "{AUTHOR_COMMIT}",\n'
            f'  "smagnet_py_sha256": "{AUTHOR_SOURCE_SHA256}",\n'
            f'  "license_sha256": "{LICENSE_SHA256}"\n'
            "}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
