"""Configuration loading helpers with no import-time file access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a ReasonForge YAML configuration is missing or invalid."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from *path* and report actionable errors."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    try:
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(content, dict):
        raise ConfigurationError(f"Top level of {config_path} must be a mapping")
    return content


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required nested mapping with a concise error on failure."""
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration key '{key}' must be a mapping")
    return value
