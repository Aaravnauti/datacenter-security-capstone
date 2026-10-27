"""Configuration loading and path helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML config. DCSEC_DATA_SOURCE overrides data.source."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    override = os.environ.get("DCSEC_DATA_SOURCE")
    if override:
        cfg["data"]["source"] = override
    return cfg


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path against the repository root unless it is absolute."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
