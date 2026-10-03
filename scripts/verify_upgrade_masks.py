"""Run pure synthetic smoke checks and print provenance for upgrade mask tools.

No project archive, labels, model outputs, or torch tensors are accessed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from flood_reliability.upgrade_masks import block_retention_mask, irregular_retention_mask, train_local_plan


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    valid = np.ones((256, 256), dtype=bool)
    blocks = [block_retention_mask(valid, "synthetic_verify", level, 20261013,
                                   namespace="eval-block-v1")
              for level in (0, 25, 50, 75, 100)]
    if not all(np.all(~left | right) for left, right in zip(blocks, blocks[1:])):
        raise RuntimeError("block masks are not nested")
    irregular = irregular_retention_mask(valid, "synthetic_verify", 0.5)
    plan = next(train_local_plan(20261013, 1, f"verify_{i}", whole_dropped=False)
                for i in range(4096)
                if train_local_plan(20261013, 1, f"verify_{i}", whole_dropped=False).apply)
    print({
        "purpose": "pure synthetic upgrade-mask verification",
        "block_pixels": [int(mask.sum()) for mask in blocks],
        "irregular_retained_pixels": int(irregular.sum()),
        "local_target_blocks": plan.target_blocks,
        "source_sha256": sha256(ROOT / "src" / "flood_reliability" / "upgrade_masks.py"),
        "test_sha256": sha256(ROOT / "tests" / "test_upgrade_masks.py"),
        "script_sha256": sha256(Path(__file__)),
    })


if __name__ == "__main__":
    main()
