import pytest
from pydantic import ValidationError

from reasonforge.dataset import (
    SYSTEM_PROMPT,
    DatasetRecord,
    deterministic_split,
    extract_gsm8k_answer,
    format_record,
)


def test_extract_gsm8k_answer_safely() -> None:
    answer = "We compute 1,200 / 2.\n#### 600"
    assert extract_gsm8k_answer(answer) == "600"
    with pytest.raises(ValueError, match="####"):
        extract_gsm8k_answer("The answer is 600")


def test_dataset_formatting_preserves_source_and_prompt() -> None:
    record = format_record(
        {"question": "What is 12 * 5?", "answer": "Multiply.\n#### 60"},
        source_split="train",
        source_index=7,
    )
    assert record["question"] == "What is 12 * 5?"
    assert record["reference_answer"] == "60"
    assert record["source_index"] == 7
    assert 1 <= record["curriculum_stage"] <= 4
    assert record["prompt"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 12 * 5?"},
    ]


def test_deterministic_split_is_repeatable_and_disjoint() -> None:
    records = list(range(100))
    train_a, validation_a = deterministic_split(records, 0.2, 17)
    train_b, validation_b = deterministic_split(records, 0.2, 17)
    assert (train_a, validation_a) == (train_b, validation_b)
    assert set(train_a).isdisjoint(validation_a)
    assert sorted(train_a + validation_a) == records


def test_different_seed_changes_validation_membership() -> None:
    records = list(range(50))
    assert deterministic_split(records, 0.2, 1)[1] != deterministic_split(records, 0.2, 2)[1]


def test_dataset_record_rejects_invalid_chat_roles() -> None:
    with pytest.raises(ValidationError):
        DatasetRecord(
            prompt=[
                {"role": "user", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "2 + 2?"},
            ],
            question="2 + 2?",
            reference_answer="4",
            source_split="train",
            source_index=0,
            curriculum_stage=1,
            curriculum_name="simple_arithmetic",
        )
