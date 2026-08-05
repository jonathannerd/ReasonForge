import json
from pathlib import Path

from reasonforge.reanalysis import classify_row, reanalyze_v1


def test_classification_separates_math_from_format() -> None:
    row = {
        "model": "base",
        "response": "The final answer is 60.",
        "reference_answer": "60",
        "answer_correct": False,
    }
    analyzed = classify_row(row)
    assert analyzed["math_accuracy"]
    assert not analyzed["strict_end_to_end_accuracy"]
    assert analyzed["formatting_or_schema_rejected_correct_answer"]


def test_reanalysis_preserves_input_and_writes_separate_results(tmp_path: Path) -> None:
    source = tmp_path / "v1.jsonl"
    original = (
        json.dumps(
            {
                "model": "aligned",
                "response": '{"method": "x", "calculations": [{"expression": "2+2", "result": "4"}], "final_answer": "4"}',
                "reference_answer": "4",
                "answer_correct": True,
            }
        )
        + "\n"
    )
    source.write_text(original, encoding="utf-8")
    destination = reanalyze_v1(source, tmp_path / "diagnostics")
    assert source.read_text(encoding="utf-8") == original
    report = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert report["source_rows"] == 1
    assert report["regenerated_completions"] is False
    assert report["metrics"]["aligned"]["math_correct"] == 1
