import json
from pathlib import Path

from reasonforge.evaluate import aggregate_metrics, update_readme_results, wilson_interval


def test_aggregate_metrics_and_failures() -> None:
    rows = [
        {
            "answer_correct": True,
            "json_valid": True,
            "schema_valid": True,
            "calculation_validity_rate": 1.0,
            "final_consistent": True,
            "reward_total": 10.0,
            "response_length": 100,
            "failure_categories": [],
        },
        {
            "answer_correct": False,
            "json_valid": False,
            "schema_valid": False,
            "calculation_validity_rate": 0.0,
            "final_consistent": False,
            "reward_total": 0.0,
            "response_length": 20,
            "failure_categories": ["invalid_json", "wrong_answer"],
        },
    ]
    metrics = aggregate_metrics(rows)
    assert metrics["final_answer_accuracy"] == 50.0
    assert metrics["math_accuracy"] == 50.0
    assert metrics["average_total_reward"] == 5.0
    assert metrics["average_response_length"] == 60
    assert metrics["failure_categories"] == {
        "correct_math_non_exact_json": 1,
        "mathematically_wrong": 1,
    }


def test_update_readme_uses_only_supplied_real_metrics(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n<!-- RESULTS:START -->\nplaceholder\n<!-- RESULTS:END -->\nafter\n",
        encoding="utf-8",
    )
    metrics = {
        "base": {
            "math_accuracy": 10.0,
            "strict_end_to_end_accuracy": 5.0,
            "json_validity_percentage": 20.0,
            "schema_compliance_percentage": 30.0,
            "calculation_validity_rate": 40.0,
            "final_consistency_percentage": 50.0,
            "truncation_rate": 2.0,
            "average_total_reward": 1.25,
        }
    }
    update_readme_results(readme, metrics)
    updated = readme.read_text(encoding="utf-8")
    assert "| Base | 10.0% | 5.0% | 20.0% | 30.0% | 40.0% | 50.0% | 2.0% | 1.250 |" in updated
    assert updated.startswith("before") and updated.endswith("after\n")
    assert json.dumps(metrics) not in updated


def test_wilson_interval_is_bounded_and_non_degenerate() -> None:
    interval = wilson_interval(64, 128)
    assert 40 < interval["lower_percentage"] < 50
    assert 50 < interval["upper_percentage"] < 60
    assert wilson_interval(0, 128)["lower_percentage"] == 0.0
