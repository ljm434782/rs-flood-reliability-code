"""Small reproducibility helpers used by later experiment CLIs."""

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON object, rejecting a missing or invalid configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a JSON object")
    return config


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure a named project logger without adding duplicate handlers."""
    logger = logging.getLogger("flood_reliability")
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def seed_everything(seed: int) -> None:
    """Set standard Python and NumPy seeds; set PyTorch seed when installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    """Hash an immutable source file in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
