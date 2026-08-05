"""Reanalyze saved v1 rows with independent math and formatting diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reasonforge.verifier import likely_truncated_text, verify_completion


def classify_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Add v2 diagnostics to one immutable v1 evaluation record."""
    response = str(row.get("response", ""))
    reference = str(row.get("reference_answer", ""))
    likely_truncated = likely_truncated_text(response)
    verification = verify_completion(response, reference, truncated=likely_truncated)
    old_credited = bool(row.get("answer_correct", False))
    parsing_failed = not verification.parse.json_valid
    schema_failed = verification.parse.json_valid and not verification.parse.schema_valid
    formatting_rejected_correct = verification.math_accuracy and not old_credited
    valid_calculations_wrong_final = bool(
        verification.calculation_checks
        and verification.calculation_validity_rate == 1.0
        and not verification.math_accuracy
    )
    if likely_truncated:
        primary = "likely_truncated"
    elif formatting_rejected_correct:
        primary = "formatting_or_schema_rejected_correct_answer"
    elif parsing_failed:
        primary = "parsing_failed"
    elif schema_failed:
        primary = "schema_failed"
    elif valid_calculations_wrong_final:
        primary = "valid_calculations_incorrect_final"
    elif not verification.math_accuracy:
        primary = "mathematically_wrong"
    else:
        primary = "mathematically_correct"
    return {
        **dict(row),
        "v1_answer_credited": old_credited,
        "extracted_answer": verification.extraction.raw_answer,
        "normalized_extracted_answer": verification.extraction.normalized_answer,
        "answer_extraction_source": verification.extraction.source,
        "math_accuracy": verification.math_accuracy,
        "json_validity": verification.parse.json_valid,
        "exact_json": verification.parse.exact_json,
        "schema_compliance": verification.parse.schema_valid,
        "calculation_validity_rate_v2": verification.calculation_validity_rate,
        "final_consistency_v2": verification.final_consistent,
        "strict_end_to_end_accuracy": verification.strict_end_to_end,
        "likely_truncated": likely_truncated,
        "parsing_failed": parsing_failed,
        "schema_failed": schema_failed,
        "formatting_or_schema_rejected_correct_answer": formatting_rejected_correct,
        "valid_calculations_incorrect_final": valid_calculations_wrong_final,
        "primary_diagnostic": primary,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate transparent counts and rates overall and per saved model."""
    if not rows:
        raise ValueError("Cannot summarize an empty v1 record set")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("model", "unknown"))].append(row)
    grouped["overall"] = list(rows)
    output: dict[str, Any] = {}
    for model, values in sorted(grouped.items()):
        count = len(values)

        def tally(key: str, group_values: Sequence[Mapping[str, Any]] = values) -> int:
            return sum(bool(value.get(key, False)) for value in group_values)

        output[model] = {
            "examples": count,
            "math_correct": tally("math_accuracy"),
            "math_accuracy_percentage": 100.0 * tally("math_accuracy") / count,
            "strict_end_to_end_correct": tally("strict_end_to_end_accuracy"),
            "strict_end_to_end_percentage": 100.0 * tally("strict_end_to_end_accuracy") / count,
            "json_valid": tally("json_validity"),
            "schema_compliant": tally("schema_compliance"),
            "likely_truncated": tally("likely_truncated"),
            "formatting_or_schema_rejected_correct_answer": tally(
                "formatting_or_schema_rejected_correct_answer"
            ),
            "parsing_failed": tally("parsing_failed"),
            "schema_failed": tally("schema_failed"),
            "valid_calculations_incorrect_final": tally("valid_calculations_incorrect_final"),
            "mathematically_wrong": count - tally("math_accuracy"),
            "primary_diagnostics": dict(
                sorted(Counter(str(value["primary_diagnostic"]) for value in values).items())
            ),
        }
    return output


def reanalyze_v1(input_path: str | Path, output_dir: str | Path) -> Path:
    """Read v1 JSONL unchanged and write separate v2 diagnostic artifacts."""
    source = Path(input_path)
    destination = Path(output_dir)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    analyzed = [classify_row(row) for row in rows]
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "per_example_reanalysis.jsonl").open("w", encoding="utf-8") as handle:
        for row in analyzed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "source_file": str(source.as_posix()),
        "source_rows": len(rows),
        "regenerated_completions": False,
        "truncation_note": (
            "v1 rows did not save token-level finish reasons; likely_truncated is a conservative "
            "text-structure heuristic and is reported separately from exact v2 finish metadata."
        ),
        "metrics": summarize(analyzed),
    }
    (destination / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="results/first-gpu-run/per_example.jsonl", help="Saved v1 JSONL"
    )
    parser.add_argument("--output-dir", default="results/v1-reanalysis")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reanalyze_v1(args.input, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Reanalysis failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
