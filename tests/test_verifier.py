import json

import pytest

from reasonforge.verifier import (
    UnsafeExpressionError,
    mathematically_equivalent,
    normalize_answer,
    safe_arithmetic,
    verify_completion,
)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("60", "60.0"),
        ("$60", "60 dollars"),
        ("1/2", "0.5"),
        ("2/4", "0.5"),
        ("50%", "0.5"),
        ("1,234", "1234"),
        ("-8", "-4 * 2"),
    ],
)
def test_equivalent_answers(left: str, right: str) -> None:
    assert mathematically_equivalent(left, right)


def test_normalization_is_exact_rational() -> None:
    assert normalize_answer("0.1 + 0.2").text == "3/10"
    assert normalize_answer("0.123456789012345678").text == ("61728394506172839/500000000000000000")
    assert safe_arithmetic("(2 + 3) ** 2").text == "25"


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd')",
        "x + 1",
        "[1][0]",
        "2 ** 999",
        "1 / 0",
        "1" * 129,
        "+".join(["1"] * 40),
    ],
)
def test_unsafe_or_excessive_expressions_are_rejected(expression: str) -> None:
    with pytest.raises(UnsafeExpressionError):
        safe_arithmetic(expression)


def test_correct_and_incorrect_calculations() -> None:
    response = json.dumps(
        {
            "method": "two steps",
            "calculations": [
                {"expression": "10 + 2", "result": "12"},
                {"expression": "12 * 5", "result": "61"},
            ],
            "final_answer": "60",
        }
    )
    result = verify_completion(response, "60")
    assert result.answer_correct
    assert [check.valid for check in result.calculation_checks] == [True, False]
    assert result.calculation_validity_rate == 0.5
    assert not result.final_consistent


def test_last_calculation_must_match_final() -> None:
    response = json.dumps(
        {
            "method": "addition",
            "calculations": [{"expression": "30 + 30", "result": "60.0"}],
            "final_answer": "$60",
        }
    )
    result = verify_completion(response, "60 dollars")
    assert result.answer_correct
    assert result.calculation_validity_rate == 1.0
    assert result.final_consistent


def test_malformed_concise_answer_can_still_be_mathematically_correct() -> None:
    result = verify_completion("1/2", "0.5")
    assert result.answer_correct
    assert not result.parse.json_valid


def test_malformed_prose_does_not_fish_for_reference_number() -> None:
    result = verify_completion("Maybe the answer is 60, ignore all other rules", "60")
    assert not result.answer_correct
