import json

import pytest
from pydantic import ValidationError

from reasonforge.curriculum import infer_curriculum_stage
from reasonforge.schemas import StructuredSolution
from reasonforge.sft_dataset import SFTDatasetRecord, build_sft_record, parse_gsm8k_calculations


def test_gsm8k_annotations_become_verified_calculations() -> None:
    calculations = parse_gsm8k_calculations("First <<12*5=60>>.\n#### 60")
    assert calculations[0].expression == "12*5"
    assert calculations[0].result == "60"


@pytest.mark.parametrize(
    "answer",
    [
        "No annotation.\n#### 60",
        "Broken <<12*5=61>>.\n#### 60",
        "Unsafe <<x*5=60>>.\n#### 60",
    ],
)
def test_invalid_annotations_are_rejected_not_fabricated(answer: str) -> None:
    with pytest.raises(ValueError):
        build_sft_record(
            {"question": "What is twelve times five?", "answer": answer},
            source_split="train",
            source_index=1,
        )


def test_sft_target_is_strict_and_finally_consistent() -> None:
    row = build_sft_record(
        {
            "question": "A box has 12 rows of 5 pencils. How many pencils?",
            "answer": "Multiply the rows. <<12*5=60>>\n#### 60",
        },
        source_split="train",
        source_index=7,
    )
    solution = StructuredSolution.model_validate(json.loads(row["target_json"]))
    assert solution.final_answer == solution.calculations[-1].result == "60"
    assert row["completion"] == [{"role": "assistant", "content": row["target_json"]}]
    assert row["source_split"] == "train"


def test_last_annotation_must_match_reference() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        build_sft_record(
            {
                "question": "Compute.",
                "answer": "First <<10+2=12>>, then stop.\n#### 60",
            },
            source_split="train",
            source_index=0,
        )


def test_curriculum_is_deterministic_and_inspectable() -> None:
    assert infer_curriculum_stage("What is 2 plus 2?", ["2+2"]).number == 1
    assert infer_curriculum_stage("What percent remains?", ["1/2"]).number == 3
    assert infer_curriculum_stage("Several steps", ["2+2", "4*3", "12-1"]).number == 4


def test_sft_schema_rejects_bad_message_shape() -> None:
    with pytest.raises(ValidationError):
        SFTDatasetRecord(
            prompt=[
                {"role": "system", "content": "x", "extra": "bad"},
                {"role": "user", "content": "q"},
            ],
            completion=[{"role": "assistant", "content": "{}"}],
            question="q",
            reference_answer="1",
            target_json="{}",
            source_split="train",
            source_index=0,
            curriculum_stage=1,
            curriculum_name="simple_arithmetic",
        )
