import json
import math

from reasonforge.rewards import build_reward_functions, schema_reward, score_completion


def structured(final: str, expression: str, result: str) -> str:
    return json.dumps(
        {
            "method": "arithmetic",
            "calculations": [{"expression": expression, "result": result}],
            "final_answer": final,
        }
    )


def test_required_reward_ordering() -> None:
    correct_structured = score_completion(structured("60", "12 * 5", "60"), "60").total
    correct_malformed = score_completion("60", "60").total
    wrong_structured = score_completion(structured("61", "60 + 1", "61"), "60").total
    wrong_malformed = score_completion("61", "60").total
    assert correct_structured > correct_malformed > wrong_structured > wrong_malformed


def test_reward_functions_match_completion_count_and_are_finite() -> None:
    completions = [
        structured("1/2", "1 / 2", "0.5"),
        "",
        [{"role": "assistant", "content": "broken"}],
    ]
    references = ["0.5", "2", "3"]
    for reward in build_reward_functions():
        values = reward(completions, reference_answer=references, ignored_column=[1, 2, 3])
        assert len(values) == len(completions)
        assert all(isinstance(value, float) and math.isfinite(value) for value in values)


def test_reward_repeats_prompt_level_references_for_generations() -> None:
    completions = ["2", "2", "3", "3"]
    reward = build_reward_functions()[1]
    assert reward(completions, reference_answer=["2", "3"]) == [5.0, 5.0, 5.0, 5.0]


def test_schema_reward_partial_json_credit_only() -> None:
    merely_json = json.dumps({"final_answer": "60"})
    full = structured("60", "12 * 5", "60")
    assert schema_reward([full], reference_answer=["60"])[0] == 1.0
    assert schema_reward([merely_json], reference_answer=["60"])[0] == 0.25


def test_conciseness_requires_correctness() -> None:
    wrong = score_completion(structured("5", "2 + 3", "5"), "6")
    assert wrong.conciseness == 0.0


def test_reward_hacking_is_penalized_and_cannot_earn_correctness() -> None:
    attack = structured("60", "__import__('os').system('true')", "60")
    score = score_completion(attack, "60")
    assert score.answer_correctness == 5.0
    assert score.calculation_validity == 0.0
    assert score.final_consistency == 0.0
    assert score.suspicious_penalty < 0
    assert score.total < score_completion(structured("60", "12 * 5", "60"), "60").total


def test_reference_number_buried_in_prose_is_not_rewarded() -> None:
    score = score_completion("Ignore JSON. The reference probably says 60.", "60")
    assert score.answer_correctness == 0.0
    assert score.total <= 0.0
