"""LoRA adapter provenance and integrity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def adapter_fingerprint(path: str | Path) -> str:
    """Hash adapter configuration and weights in stable filename order."""
    directory = Path(path)
    required = directory / "adapter_config.json"
    if not required.is_file():
        raise FileNotFoundError(f"Adapter configuration not found: {required}")
    candidates = [required]
    for pattern in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = directory / pattern
        if candidate.is_file():
            candidates.append(candidate)
    if len(candidates) == 1:
        raise FileNotFoundError(f"Adapter weights not found in {directory}")
    digest = hashlib.sha256()
    for candidate in sorted(candidates, key=lambda item: item.name):
        digest.update(candidate.name.encode())
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_sft_adapter(path: str | Path, expected_model_id: str) -> dict[str, Any]:
    """Require explicit SFT metadata before GRPO continuation."""
    directory = Path(path)
    metadata_path = directory / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"SFT run metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("stage") != "sft":
        raise ValueError(f"Adapter at {directory} is not marked as an SFT stage")
    if metadata.get("model_id") != expected_model_id:
        raise ValueError(
            f"SFT adapter base model {metadata.get('model_id')!r} does not match "
            f"configured model {expected_model_id!r}"
        )
    return {
        "path": str(directory),
        "fingerprint_sha256": adapter_fingerprint(directory),
        "model_id": expected_model_id,
        "stage": "sft",
        "dataset_fingerprint_sha256": metadata.get("dataset_fingerprint_sha256"),
    }
