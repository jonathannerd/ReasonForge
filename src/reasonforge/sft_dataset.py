"""Leakage-safe GSM8K conversion into validated structured SFT targets."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reasonforge.config import ConfigurationError
from reasonforge.curriculum import infer_curriculum_stage
from reasonforge.dataset import SYSTEM_PROMPT, extract_gsm8k_answer
from reasonforge.schemas import Calculation, StructuredSolution
from reasonforge.verifier import (
    UnsafeExpressionError,
    mathematically_equivalent,
    normalize_answer,
    safe_arithmetic,
)

_ANNOTATION_RE = re.compile(r"<<([^<>]+)>>")
_COMMA_BETWEEN_DIGITS_RE = re.compile(r"(?<=\d),(?=\d)")


class SFTDatasetRecord(BaseModel):
    """Validated prompt-completion record with auditable source provenance."""

    model_config = ConfigDict(extra="forbid")

    prompt: list[dict[str, str]] = Field(min_length=2, max_length=2)
    completion: list[dict[str, str]] = Field(min_length=1, max_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    target_json: str = Field(min_length=1)
    source: str = "gsm8k"
    source_split: str
    source_index: int = Field(ge=0)
    curriculum_stage: int = Field(ge=1, le=4)
    curriculum_name: str = Field(min_length=1)

    @field_validator("prompt", "completion")
    @classmethod
    def messages_have_exact_keys(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        if any(set(message) != {"role", "content"} for message in value):
            raise ValueError("messages must contain exactly role and content")
        return value


@dataclass(frozen=True)
class PreparedSFTData:
    train: Any
    validation: Any
    manifest: dict[str, Any]


def parse_gsm8k_calculations(answer: str) -> list[Calculation]:
    """Parse only verified ``<<expression=result>>`` GSM8K annotations."""
    calculations: list[Calculation] = []
    for annotation in _ANNOTATION_RE.findall(answer):
        if "=" not in annotation:
            raise ValueError("calculation annotation is missing '='")
        expression, claimed = (part.strip() for part in annotation.rsplit("=", 1))
        expression = _COMMA_BETWEEN_DIGITS_RE.sub("", expression)
        claimed = _COMMA_BETWEEN_DIGITS_RE.sub("", claimed)
        try:
            expression_value = safe_arithmetic(expression)
            claimed_value = normalize_answer(claimed)
        except UnsafeExpressionError as exc:
            raise ValueError(f"unsupported calculation annotation: {exc}") from exc
        if expression_value.value != claimed_value.value:
            raise ValueError("calculation annotation has an incorrect claimed result")
        calculations.append(Calculation(expression=expression, result=claimed_value.text))
    if not calculations:
        raise ValueError("worked answer contains no verifiable calculation annotations")
    return calculations


def build_sft_record(
    example: Mapping[str, Any], *, source_split: str, source_index: int
) -> dict[str, Any]:
    """Create one strict target, rejecting rather than fabricating invalid work."""
    question = str(example.get("question", "")).strip()
    raw_answer = str(example.get("answer", ""))
    reference = extract_gsm8k_answer(raw_answer)
    calculations = parse_gsm8k_calculations(raw_answer)
    if not mathematically_equivalent(calculations[-1].result, reference):
        raise ValueError("last verified annotation is inconsistent with the final answer")
    stage = infer_curriculum_stage(question, (item.expression for item in calculations))
    solution = StructuredSolution(
        method="single-step arithmetic" if len(calculations) == 1 else "multi-step arithmetic",
        calculations=calculations,
        final_answer=reference,
    )
    target = json.dumps(solution.model_dump(), separators=(",", ":"), ensure_ascii=False)
    return SFTDatasetRecord(
        prompt=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        completion=[{"role": "assistant", "content": target}],
        question=question,
        reference_answer=reference,
        target_json=target,
        source_split=source_split,
        source_index=source_index,
        curriculum_stage=stage.number,
        curriculum_name=stage.name,
    ).model_dump()


def _fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = {
            key: row[key]
            for key in (
                "source",
                "source_split",
                "source_index",
                "question",
                "reference_answer",
                "target_json",
                "curriculum_stage",
            )
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _convert(
    dataset: Any,
    *,
    split_name: str,
    requested_size: int,
    max_stage: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for example in dataset:
        try:
            row = build_sft_record(
                example,
                source_split="train",
                source_index=int(example["_source_index"]),
            )
        except ValueError as exc:
            rejected[str(exc)] += 1
            continue
        if int(row["curriculum_stage"]) > max_stage:
            rejected["above_configured_curriculum_stage"] += 1
            continue
        rows.append(row)
        if len(rows) >= requested_size:
            break
    if not rows:
        raise ValueError(f"No valid {split_name} SFT records remained after validation")
    rows.sort(key=lambda row: (int(row["curriculum_stage"]), int(row["source_index"])))
    return rows, rejected


def prepare_sft_datasets(config: Mapping[str, Any]) -> PreparedSFTData:
    """Load only GSM8K train data and return disjoint SFT train/validation sets."""
    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install ReasonForge training dependencies to prepare SFT data") from exc
    dataset_config = config.get("dataset")
    sft_config = config.get("sft")
    curriculum = config.get("curriculum", {})
    if not isinstance(dataset_config, Mapping) or not isinstance(sft_config, Mapping):
        raise ConfigurationError("SFT configuration requires dataset and sft mappings")
    if not isinstance(curriculum, Mapping):
        raise ConfigurationError("curriculum must be a mapping")
    seed = int(config.get("seed", 42))
    validation_fraction = float(dataset_config.get("validation_fraction", 0.1))
    if not 0.0 < validation_fraction < 1.0:
        raise ConfigurationError("dataset.validation_fraction must be between 0 and 1")
    train_size = int(sft_config.get("train_size", 1024))
    validation_size = int(sft_config.get("validation_size", 128))
    max_stage = int(curriculum.get("max_stage", 4))
    if train_size < 1 or validation_size < 1 or not 1 <= max_stage <= 4:
        raise ConfigurationError("SFT sizes must be positive and max_stage must be 1..4")
    raw = load_dataset(str(dataset_config.get("name", "openai/gsm8k")), "main")["train"]
    raw = raw.add_column("_source_index", range(len(raw))).shuffle(seed=seed)
    split = raw.train_test_split(test_size=validation_fraction, seed=seed, shuffle=True)
    train_rows, train_rejected = _convert(
        split["train"], split_name="train", requested_size=train_size, max_stage=max_stage
    )
    validation_rows, validation_rejected = _convert(
        split["test"],
        split_name="validation",
        requested_size=validation_size,
        max_stage=max_stage,
    )
    train_indices = {row["source_index"] for row in train_rows}
    validation_indices = {row["source_index"] for row in validation_rows}
    if train_indices & validation_indices:
        raise RuntimeError("SFT train/validation source indices overlap")
    combined = [*train_rows, *validation_rows]
    manifest = {
        "source_dataset": str(dataset_config.get("name", "openai/gsm8k")),
        "source_splits_used": ["train"],
        "official_test_examples_used": 0,
        "seed": seed,
        "max_curriculum_stage": max_stage,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "train_stage_counts": dict(
            sorted(Counter(row["curriculum_name"] for row in train_rows).items())
        ),
        "validation_stage_counts": dict(
            sorted(Counter(row["curriculum_name"] for row in validation_rows).items())
        ),
        "rejections": {
            "train": dict(sorted(train_rejected.items())),
            "validation": dict(sorted(validation_rejected.items())),
        },
        "fingerprint_sha256": _fingerprint(combined),
        "train_validation_overlap": 0,
    }
    return PreparedSFTData(
        train=Dataset.from_list(train_rows),
        validation=Dataset.from_list(validation_rows),
        manifest=manifest,
    )
