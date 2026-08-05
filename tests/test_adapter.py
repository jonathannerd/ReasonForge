import json
from pathlib import Path

import pytest

from reasonforge.adapter import adapter_fingerprint, validate_sft_adapter


def make_adapter(path: Path, *, stage: str = "sft", model_id: str = "model") -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    (path / "run_metadata.json").write_text(
        json.dumps({"stage": stage, "model_id": model_id, "dataset_fingerprint_sha256": "data"}),
        encoding="utf-8",
    )


def test_adapter_fingerprint_and_sft_assertion(tmp_path: Path) -> None:
    path = tmp_path / "adapter"
    make_adapter(path)
    assert len(adapter_fingerprint(path)) == 64
    validated = validate_sft_adapter(path, "model")
    assert validated["stage"] == "sft"
    assert validated["dataset_fingerprint_sha256"] == "data"


def test_grpo_adapter_cannot_masquerade_as_sft(tmp_path: Path) -> None:
    path = tmp_path / "adapter"
    make_adapter(path, stage="grpo")
    with pytest.raises(ValueError, match="not marked as an SFT"):
        validate_sft_adapter(path, "model")
