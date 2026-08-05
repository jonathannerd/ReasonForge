import json

from reasonforge.rewards import RewardDiagnosticsCollector, build_reward_functions


def structured(answer: str) -> str:
    return json.dumps(
        {
            "method": "answer",
            "calculations": [{"expression": f"{answer} * 1", "result": answer}],
            "final_answer": answer,
        }
    )


def test_group_diagnostics_capture_signal_and_zero_variance() -> None:
    collector = RewardDiagnosticsCollector(num_generations=2)
    rewards = build_reward_functions(diagnostics=collector)
    completions = [structured("4"), structured("5"), "bad", "also bad"]
    references = ["4", "6"]
    for reward in rewards:
        reward(completions, reference_answer=references)
    summary = collector.summary()
    assert summary["groups"] == 2
    assert summary["groups_with_any_correct"] == 1
    assert summary["groups_with_any_correct_percentage"] == 50.0
    assert summary["all_incorrect_groups"] == 1
    assert summary["all_incorrect_group_percentage"] == 50.0
    assert summary["zero_correctness_variance_groups"] == 1
    assert summary["correct_completion_rate"] == 25.0
    assert summary["mean_correctness_reward"] == 1.25
    assert summary["mean_group_total_reward_stddev"] > 0
    assert summary["truncated_completion_rate"] == 0.0
    assert summary["mean_unique_generations"] == 2
