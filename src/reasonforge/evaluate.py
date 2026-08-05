"""Paired held-out evaluation for base, SFT, and SFT-plus-GRPO policies."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from reasonforge.config import ConfigurationError, load_yaml, require_mapping
from reasonforge.dataset import prepare_datasets
from reasonforge.inference import GenerationSettings, LazyModelRunner
from reasonforge.plotting import plot_comparison, plot_failures, plot_reward_components
from reasonforge.rewards import score_completion
from reasonforge.verifier import verify_completion

LOGGER = logging.getLogger(__name__)

MODEL_DISPLAY_NAMES = {"base": "Base", "sft": "SFT", "sft_grpo": "SFT + GRPO"}


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> dict[str, float]:
    """Return a two-sided 95% Wilson interval as percentages."""
    if count < 1 or not 0 <= successes <= count:
        raise ValueError("Wilson interval requires 0 <= successes <= count and count > 0")
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return {
        "lower_percentage": 100.0 * max(0.0, center - margin),
        "upper_percentage": 100.0 * min(1.0, center + margin),
        "confidence_level": 0.95,
        "method": "Wilson score interval",
    }


def _primary_failure(row: Mapping[str, Any]) -> str:
    if bool(row.get("truncated", False)):
        return "truncated"
    if not bool(row.get("math_accuracy", row.get("answer_correct", False))):
        return "mathematically_wrong"
    if not bool(row.get("json_validity", row.get("json_valid", False))):
        return "correct_math_invalid_json"
    if not bool(row.get("schema_valid", False)):
        return "correct_math_schema_violation"
    if float(row.get("calculation_validity_rate", 0.0)) < 1.0:
        return "correct_math_invalid_calculation"
    if not bool(row.get("final_consistent", False)):
        return "correct_math_inconsistent_final"
    if not bool(row.get("strict_end_to_end_accuracy", False)):
        return "correct_math_non_exact_json"
    return "passed_strict_end_to_end"


def _row(
    *,
    model_name: str,
    item: Mapping[str, Any],
    response: str,
    reward_weights: Mapping[str, float] | None,
    completion_tokens: int | None = None,
    truncated: bool = False,
    finish_reason: str = "unknown",
) -> dict[str, Any]:
    verification = verify_completion(response, str(item["reference_answer"]), truncated=truncated)
    reward = score_completion(response, str(item["reference_answer"]), reward_weights)
    row = {
        "model": model_name,
        "source_split": item["source_split"],
        "source_index": item["source_index"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "response": response,
        "extracted_answer": verification.extraction.raw_answer,
        "normalized_extracted_answer": verification.extraction.normalized_answer,
        "answer_extraction_source": verification.extraction.source,
        "answer_extraction_confidence": verification.extraction.confidence,
        "math_accuracy": verification.math_accuracy,
        "strict_end_to_end_accuracy": verification.strict_end_to_end,
        "answer_correct": verification.answer_correct,
        "json_validity": verification.parse.json_valid,
        "json_valid": verification.parse.json_valid,
        "exact_json": verification.parse.exact_json,
        "schema_valid": verification.parse.schema_valid,
        "calculation_validity_rate": verification.calculation_validity_rate,
        "final_consistent": verification.final_consistent,
        "response_length": len(response),
        "completion_tokens": completion_tokens,
        "truncated": truncated,
        "finish_reason": finish_reason,
        "failure_reason": verification.failure_reason,
        "verification_failure_categories": verification.failure_categories,
        "reward_schema": reward.schema_score,
        "reward_answer_correctness": reward.answer_correctness,
        "reward_calculation_validity": reward.calculation_validity,
        "reward_final_consistency": reward.final_consistency,
        "reward_conciseness": reward.conciseness,
        "reward_suspicious_penalty": reward.suspicious_penalty,
        "reward_total": reward.total,
    }
    row["primary_failure_category"] = _primary_failure(row)
    row["failure_categories"] = verification.failure_categories
    return row


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate math, format, stability, and reward metrics for one model."""
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation")
    count = len(rows)

    def values(key: str, fallback: str | None = None) -> list[bool]:
        return [bool(row.get(key, row.get(fallback, False) if fallback else False)) for row in rows]

    def percentage(flags: Sequence[bool]) -> float:
        return 100.0 * sum(flags) / count

    math_flags = values("math_accuracy", "answer_correct")
    strict_flags = values("strict_end_to_end_accuracy")
    json_flags = values("json_validity", "json_valid")
    schema_flags = values("schema_valid")
    consistency_flags = values("final_consistent")
    truncation_flags = values("truncated")
    failures = Counter(
        str(row.get("primary_failure_category") or _primary_failure(row)) for row in rows
    )

    def average(key: str, default: float = 0.0) -> float:
        return mean(float(row.get(key, default) or 0.0) for row in rows)

    math_percentage = percentage(math_flags)
    json_percentage = percentage(json_flags)
    return {
        "examples": count,
        "math_accuracy": math_percentage,
        "math_accuracy_wilson_95": wilson_interval(sum(math_flags), count),
        "strict_end_to_end_accuracy": percentage(strict_flags),
        "strict_end_to_end_wilson_95": wilson_interval(sum(strict_flags), count),
        "json_validity_percentage": json_percentage,
        "schema_compliance_percentage": percentage(schema_flags),
        "calculation_validity_rate": 100.0
        * mean(float(row.get("calculation_validity_rate", 0.0)) for row in rows),
        "final_consistency_percentage": percentage(consistency_flags),
        "truncation_rate": percentage(truncation_flags),
        "average_total_reward": average("reward_total"),
        "average_reward_schema": average("reward_schema"),
        "average_reward_answer_correctness": average("reward_answer_correctness"),
        "average_reward_calculation_validity": average("reward_calculation_validity"),
        "average_reward_final_consistency": average("reward_final_consistency"),
        "average_reward_conciseness": average("reward_conciseness"),
        "average_reward_suspicious_penalty": average("reward_suspicious_penalty"),
        "average_response_length": average("response_length"),
        "average_completion_tokens": average("completion_tokens"),
        "failure_categories": dict(sorted(failures.items())),
        # Backward-compatible aliases retained for v1 analysis consumers.
        "final_answer_accuracy": math_percentage,
        "valid_json_percentage": json_percentage,
    }


def _write_artifacts(
    rows: list[dict[str, Any]], output_dir: Path, *, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Pandas is required to write evaluation artifacts") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            **row,
            "failure_categories": json.dumps(row["failure_categories"]),
            "verification_failure_categories": json.dumps(row["verification_failure_categories"]),
        }
        for row in rows
    ]
    csv_serializable = [
        {
            key: value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
            if isinstance(value, str)
            else value
            for key, value in row.items()
        }
        for row in serializable
    ]
    pd.DataFrame(csv_serializable).to_csv(output_dir / "per_example.csv", index=False)
    with (output_dir / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    grouped = {
        model_name: aggregate_metrics([row for row in rows if row["model"] == model_name])
        for model_name in sorted({str(row["model"]) for row in rows})
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(grouped, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    examples = {
        model_name: {
            "math_correct": [
                row for row in rows if row["model"] == model_name and row["math_accuracy"]
            ][:3],
            "strict_correct": [
                row
                for row in rows
                if row["model"] == model_name and row["strict_end_to_end_accuracy"]
            ][:3],
            "failed": [
                row for row in rows if row["model"] == model_name and not row["math_accuracy"]
            ][:3],
        }
        for model_name in grouped
    }
    (output_dir / "representative_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    evaluation_manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "paired_examples": len(rows) // len(grouped),
        "models": list(grouped),
        "confidence_intervals": "two-sided 95% Wilson score intervals",
        "raw_text_authority": "per_example.jsonl",
        "csv_embedded_newlines": "escaped as the two characters \\n",
        **dict(manifest or {}),
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    plot_comparison(grouped, output_dir / "model_comparison.png")
    plot_failures(grouped, output_dir / "failure_categories.png")
    plot_reward_components(grouped, output_dir / "reward_components.png")
    return grouped


def update_readme_results(
    readme_path: str | Path, metrics: Mapping[str, Mapping[str, Any]]
) -> None:
    """Replace the README results block using supplied measured metrics only."""
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    start_marker = "<!-- RESULTS:START -->"
    end_marker = "<!-- RESULTS:END -->"
    if start_marker not in text or end_marker not in text:
        raise ValueError("README is missing generated-results markers")
    header = (
        "| Model | Math accuracy | Strict E2E | JSON valid | Schema | Calc validity | "
        "Consistency | Truncated | Avg reward |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for model_name, values in metrics.items():
        lines.append(
            "| {name} | {math_accuracy:.1f}% | {strict_end_to_end_accuracy:.1f}% | "
            "{json_validity_percentage:.1f}% | {schema_compliance_percentage:.1f}% | "
            "{calculation_validity_rate:.1f}% | {final_consistency_percentage:.1f}% | "
            "{truncation_rate:.1f}% | {average_total_reward:.3f} |".format(
                name=MODEL_DISPLAY_NAMES.get(model_name, model_name), **values
            )
        )
    replacement = start_marker + "\n" + "\n".join(lines) + "\n" + end_marker
    before, remainder = text.split(start_marker, 1)
    _, after = remainder.split(end_marker, 1)
    path.write_text(before + replacement + after, encoding="utf-8")


def _model_specs(evaluation: Mapping[str, Any]) -> dict[str, Path | None]:
    configured = evaluation.get("models")
    if isinstance(configured, Mapping):
        specs: dict[str, Path | None] = {}
        for name, value in configured.items():
            if value is None:
                specs[str(name)] = None
            elif isinstance(value, Mapping):
                path = value.get("adapter_path")
                specs[str(name)] = Path(str(path)) if path else None
            else:
                specs[str(name)] = Path(str(value))
    else:
        specs = {
            "base": None,
            "aligned": Path(str(evaluation.get("adapter_path", "outputs/reasonforge-adapter"))),
        }
    if len(specs) < 2:
        raise ConfigurationError("evaluation.models must configure at least two models")
    for name, path in specs.items():
        if path is not None and not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Adapter for {name!r} not found at {path}")
    return specs


def evaluate(config: Mapping[str, Any], *, update_readme: bool = False) -> Path:
    """Run paired generation on the same official GSM8K test examples."""
    model_config = require_mapping(dict(config), "model")
    evaluation = require_mapping(dict(config), "evaluation")
    model_id = str(model_config.get("id", "Qwen/Qwen2.5-0.5B-Instruct"))
    specs = _model_specs(evaluation)
    settings = GenerationSettings(
        max_new_tokens=int(evaluation.get("max_new_tokens", 256)),
        temperature=float(evaluation.get("temperature", 0.0)),
        top_p=float(evaluation.get("top_p", 0.95)),
        seed=int(config.get("seed", 42)),
    )
    dataset = prepare_datasets(config)["test"]
    reward_weights = config.get("rewards")
    if not isinstance(reward_weights, Mapping):
        reward_weights = None
    rows: list[dict[str, Any]] = []
    for model_name, model_adapter in specs.items():
        runner = LazyModelRunner(model_id, model_adapter)
        LOGGER.info("Evaluating %s on %d held-out examples", model_name, len(dataset))
        for item in dataset:
            generated = runner.generate_result(str(item["question"]), settings)
            rows.append(
                _row(
                    model_name=model_name,
                    item=item,
                    response=generated.text,
                    reward_weights=reward_weights,
                    completion_tokens=generated.completion_tokens,
                    truncated=generated.truncated,
                    finish_reason=generated.finish_reason,
                )
            )
        del runner
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    output_dir = Path(str(evaluation.get("output_dir", "results/latest")))
    metrics = _write_artifacts(
        rows,
        output_dir,
        manifest={
            "model_id": model_id,
            "model_adapters": {name: str(path) if path else None for name, path in specs.items()},
            "source_dataset": str(config.get("dataset", {}).get("name", "openai/gsm8k")),
            "source_split": "test",
            "generation_settings": settings.__dict__,
        },
    )
    if update_readme:
        update_readme_results("README.md", metrics)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation_v2.yaml", help="Evaluation YAML")
    parser.add_argument(
        "--update-readme", action="store_true", help="Insert measured metrics into README markers"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        output = evaluate(load_yaml(args.config), update_readme=args.update_readme)
    except (ConfigurationError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.error("Evaluation failed: %s", exc)
        return 2
    LOGGER.info("Saved evaluation artifacts to %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
