import json
from pathlib import Path

from reasonforge.evaluate import aggregate_metrics, update_readme_results


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
    assert metrics["average_total_reward"] == 5.0
    assert metrics["average_response_length"] == 60
    assert metrics["failure_categories"] == {"invalid_json": 1, "wrong_answer": 1}


def test_update_readme_uses_only_supplied_real_metrics(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n<!-- RESULTS:START -->\nplaceholder\n<!-- RESULTS:END -->\nafter\n",
        encoding="utf-8",
    )
    metrics = {
        "base": {
            "final_answer_accuracy": 10.0,
            "valid_json_percentage": 20.0,
            "schema_compliance_percentage": 30.0,
            "calculation_validity_rate": 40.0,
            "final_consistency_percentage": 50.0,
            "average_total_reward": 1.25,
        }
    }
    update_readme_results(readme, metrics)
    updated = readme.read_text(encoding="utf-8")
    assert "| base | 10.0% | 20.0% | 30.0% | 40.0% | 50.0% | 1.250 |" in updated
    assert updated.startswith("before") and updated.endswith("after\n")
    assert json.dumps(metrics) not in updated
