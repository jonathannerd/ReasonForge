import json

from reasonforge.parsing import completion_to_text, parse_completion

VALID = {
    "method": "multiplication",
    "calculations": [{"expression": "12 * 5", "result": "60"}],
    "final_answer": "60",
}


def test_valid_json_and_schema() -> None:
    result = parse_completion(json.dumps(VALID))
    assert result.json_valid and result.schema_valid and result.exact_json
    assert result.solution is not None
    assert result.solution.final_answer == "60"


def test_fenced_json_is_valid() -> None:
    result = parse_completion(f"```json\n{json.dumps(VALID)}\n```")
    assert result.json_valid and result.schema_valid
    assert not result.exact_json


def test_surrounding_prose_loses_exactness() -> None:
    result = parse_completion(f"Here: {json.dumps(VALID)}")
    assert result.schema_valid
    assert not result.exact_json


def test_invalid_json_and_empty_completion() -> None:
    assert not parse_completion("{not json}").json_valid
    assert parse_completion([]).error == "completion is empty"


def test_missing_field_and_extra_field_fail_schema() -> None:
    missing = {"method": "add", "calculations": []}
    extra = {**VALID, "reasoning": "trust me"}
    assert parse_completion(json.dumps(missing)).json_valid
    assert not parse_completion(json.dumps(missing)).schema_valid
    assert not parse_completion(json.dumps(extra)).schema_valid


def test_trl_conversational_completion_structure() -> None:
    completion = [{"role": "assistant", "content": json.dumps(VALID)}]
    assert completion_to_text(completion) == json.dumps(VALID)
    assert parse_completion(completion).schema_valid


def test_completion_size_limit() -> None:
    result = parse_completion("{" + "x" * 4_001)
    assert result.error == "completion exceeds size limit"


def test_duplicate_keys_and_nonstandard_constants_are_rejected() -> None:
    duplicate = '{"method":"x","method":"y","calculations":[],"final_answer":"1"}'
    nonstandard = '{"method":"x","calculations":[],"final_answer":NaN}'
    assert not parse_completion(duplicate).json_valid
    assert not parse_completion(nonstandard).json_valid
