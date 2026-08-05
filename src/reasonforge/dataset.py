"""Leakage-aware, deterministic GSM8K preparation for conversational GRPO."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reasonforge.config import ConfigurationError, load_yaml, require_mapping
from reasonforge.curriculum import infer_curriculum_stage
from reasonforge.verifier import UnsafeExpressionError, normalize_answer

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")

SYSTEM_PROMPT = """You solve arithmetic and introductory algebra problems. Return exactly one JSON object and no prose or Markdown. The schema is:
{"method": "short method label", "calculations": [{"expression": "restricted arithmetic using numbers and + - * / ** parentheses", "result": "numeric result"}], "final_answer": "numeric answer"}
Every calculation must be valid. The last calculation result must equal final_answer. Use no variables, functions, code, or unverified explanation."""

_GSM8K_FINAL_RE = re.compile(r"####\s*([^\n\r]+)\s*$")
_GSM8K_CALC_RE = re.compile(r"<<([^<>]+)>>")


class DatasetRecord(BaseModel):
    """Validated GRPO record with source fields preserved for auditing."""

    model_config = ConfigDict(extra="forbid")

    prompt: list[dict[str, str]] = Field(min_length=2, max_length=2)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    source: str = "gsm8k"
    source_split: str
    source_index: int = Field(ge=0)
    curriculum_stage: int = Field(ge=1, le=4)
    curriculum_name: str = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def prompt_has_system_and_user_messages(
        cls, value: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Enforce the exact two-message conversational contract."""
        expected_roles = ("system", "user")
        for message, role in zip(value, expected_roles, strict=True):
            if set(message) != {"role", "content"} or message["role"] != role:
                raise ValueError(f"prompt message must be exactly a {role!r} role/content mapping")
            if not message["content"].strip():
                raise ValueError("prompt content cannot be empty")
        return value


def extract_gsm8k_answer(answer: str) -> str:
    """Extract and normalize the final answer after GSM8K's ``####`` marker."""
    if not isinstance(answer, str):
        raise ValueError("GSM8K answer must be a string")
    match = _GSM8K_FINAL_RE.search(answer)
    if not match:
        raise ValueError("GSM8K answer is missing a final '####' marker")
    raw = match.group(1).strip()
    try:
        return normalize_answer(raw).text
    except UnsafeExpressionError as exc:
        raise ValueError(f"Unsupported GSM8K reference answer {raw!r}: {exc}") from exc


def format_record(
    example: Mapping[str, Any],
    *,
    source_split: str,
    source_index: int,
) -> dict[str, Any]:
    """Convert one raw GSM8K example into a validated conversational record."""
    question = str(example.get("question", "")).strip()
    raw_answer = str(example.get("answer", ""))
    reference = extract_gsm8k_answer(raw_answer)
    expressions = [
        annotation.rsplit("=", 1)[0].strip()
        for annotation in _GSM8K_CALC_RE.findall(raw_answer)
        if "=" in annotation
    ]
    stage = infer_curriculum_stage(question, expressions)
    record = DatasetRecord(
        prompt=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        question=question,
        reference_answer=reference,
        source_split=source_split,
        source_index=source_index,
        curriculum_stage=stage.number,
        curriculum_name=stage.name,
    )
    return record.model_dump()


def deterministic_split(
    records: Sequence[T], validation_fraction: float, seed: int
) -> tuple[list[T], list[T]]:
    """Split records deterministically without mutating input order or content."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    validation_size = max(1, round(len(indices) * validation_fraction))
    validation_indices = set(indices[:validation_size])
    train = [record for index, record in enumerate(records) if index not in validation_indices]
    validation = [record for index, record in enumerate(records) if index in validation_indices]
    return train, validation


def _take(dataset: Any, size: int | None) -> Any:
    if size is None:
        return dataset
    if not isinstance(size, int) or size < 1:
        raise ConfigurationError("dataset subset sizes must be positive integers or null")
    return dataset.select(range(min(size, len(dataset))))


def prepare_datasets(config: Mapping[str, Any]) -> dict[str, Any]:
    """Download GSM8K and return disjoint train, validation, and test datasets.

    The official test set is never sampled into training. Validation is carved
    only from the official training split before independent subset limits apply.
    """
    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install ReasonForge training dependencies to prepare GSM8K") from exc

    data_config = config.get("dataset")
    if not isinstance(data_config, Mapping):
        raise ConfigurationError("Configuration key 'dataset' must be a mapping")
    seed = int(config.get("seed", 42))
    curriculum = config.get("curriculum", {})
    if not isinstance(curriculum, Mapping):
        raise ConfigurationError("curriculum must be a mapping")
    curriculum_enabled = bool(curriculum.get("enabled", False))
    max_stage = int(curriculum.get("max_stage", 4))
    if not 1 <= max_stage <= 4:
        raise ConfigurationError("curriculum.max_stage must be between 1 and 4")
    validation_fraction = float(data_config.get("validation_fraction", 0.1))
    if not 0.0 < validation_fraction < 1.0:
        raise ConfigurationError("dataset.validation_fraction must be between 0 and 1")
    raw = load_dataset(str(data_config.get("name", "openai/gsm8k")), "main")
    train_source = raw["train"].add_column("_source_index", range(len(raw["train"])))
    train_raw = train_source.shuffle(seed=seed)
    split = train_raw.train_test_split(test_size=validation_fraction, seed=seed, shuffle=True)
    test_source = raw["test"].add_column("_source_index", range(len(raw["test"])))
    test_raw = test_source.shuffle(seed=seed)

    requested = {
        "train": data_config.get("train_size", 512),
        "validation": data_config.get("validation_size", 128),
        "test": data_config.get("test_size", 128),
    }
    sources = {
        "train": (_take(split["train"], requested["train"]), "train"),
        "validation": (_take(split["test"], requested["validation"]), "train"),
        "test": (_take(test_raw, requested["test"]), "test"),
    }
    prepared: dict[str, Any] = {}
    for split_name, (source_dataset, source_split) in sources.items():
        rows = [
            format_record(
                example,
                source_split=source_split,
                source_index=int(example["_source_index"]),
            )
            for example in source_dataset
        ]
        if curriculum_enabled and split_name == "train":
            rows = [row for row in rows if int(row["curriculum_stage"]) <= max_stage]
            rows.sort(key=lambda row: (int(row["curriculum_stage"]), int(row["source_index"])))
        prepared[split_name] = Dataset.from_list(rows)
    return prepared


def save_prepared_datasets(datasets: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Persist prepared split directories plus a human-readable manifest."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, int] = {}
    for name, dataset in datasets.items():
        dataset.save_to_disk(destination / name)
        manifest[name] = len(dataset)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml", help="Training YAML path")
    parser.add_argument("--output-dir", help="Override dataset cache output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare GSM8K only when explicitly invoked from the command line."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_yaml(args.config)
        data_config = require_mapping(config, "dataset")
        datasets = prepare_datasets(config)
        output = args.output_dir or data_config.get("prepared_path", "outputs/dataset")
        destination = save_prepared_datasets(datasets, output)
    except (ConfigurationError, RuntimeError, ValueError) as exc:
        LOGGER.error("Dataset preparation failed: %s", exc)
        return 2
    LOGGER.info("Saved prepared dataset to %s", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
