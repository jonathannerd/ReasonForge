"""Held-out, before-versus-after ReasonForge evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

from reasonforge.config import ConfigurationError, load_yaml, require_mapping
from reasonforge.dataset import prepare_datasets
from reasonforge.inference import GenerationSettings, LazyModelRunner
from reasonforge.plotting import plot_comparison
from reasonforge.rewards import score_completion
from reasonforge.verifier import verify_completion

LOGGER = logging.getLogger(__name__)


def _row(
    *,
    model_name: str,
    item: Mapping[str, Any],
    response: str,
    reward_weights: Mapping[str, float] | None,
) -> dict[str, Any]:
    verification = verify_completion(response, str(item["reference_answer"]))
    reward = score_completion(response, str(item["reference_answer"]), reward_weights)
    return {
        "model": model_name,
        "source_split": item["source_split"],
        "source_index": item["source_index"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "response": response,
        "json_valid": verification.parse.json_valid and verification.parse.exact_json,
        "schema_valid": verification.parse.schema_valid,
        "answer_correct": verification.answer_correct,
        "calculation_validity_rate": verification.calculation_validity_rate,
        "final_consistent": verification.final_consistent,
        "response_length": len(response),
        "failure_reason": verification.failure_reason,
        "failure_categories": verification.failure_categories,
        "reward_schema": reward.schema_score,
        "reward_answer_correctness": reward.answer_correctness,
        "reward_calculation_validity": reward.calculation_validity,
        "reward_final_consistency": reward.final_consistency,
        "reward_conciseness": reward.conciseness,
        "reward_suspicious_penalty": reward.suspicious_penalty,
        "reward_total": reward.total,
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate verification metrics for one model's per-example rows."""
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation")
    failures: Counter[str] = Counter()
    for row in rows:
        failures.update(str(category) for category in row["failure_categories"])
    count = len(rows)

    def percent(key: str) -> float:
        return 100.0 * sum(bool(row[key]) for row in rows) / count

    return {
        "examples": count,
        "final_answer_accuracy": percent("answer_correct"),
        "valid_json_percentage": percent("json_valid"),
        "schema_compliance_percentage": percent("schema_valid"),
        "calculation_validity_rate": 100.0
        * mean(float(row["calculation_validity_rate"]) for row in rows),
        "final_consistency_percentage": percent("final_consistent"),
        "average_total_reward": mean(float(row["reward_total"]) for row in rows),
        "average_response_length": mean(int(row["response_length"]) for row in rows),
        "failure_categories": dict(sorted(failures.items())),
    }


def _write_artifacts(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Pandas is required to write evaluation artifacts") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [
        {**row, "failure_categories": json.dumps(row["failure_categories"])} for row in rows
    ]
    pd.DataFrame(serializable).to_csv(output_dir / "per_example.csv", index=False)
    with (output_dir / "per_example.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    grouped = {
        model_name: aggregate_metrics([row for row in rows if row["model"] == model_name])
        for model_name in sorted({str(row["model"]) for row in rows})
    }
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(grouped, indent=2) + "\n", encoding="utf-8"
    )
    examples = {
        model_name: {
            "successful": [
                row for row in rows if row["model"] == model_name and row["answer_correct"]
            ][:3],
            "failed": [
                row for row in rows if row["model"] == model_name and not row["answer_correct"]
            ][:3],
        }
        for model_name in grouped
    }
    (output_dir / "representative_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_comparison(grouped, output_dir / "comparison.png")
    return grouped


def update_readme_results(
    readme_path: str | Path, metrics: Mapping[str, Mapping[str, Any]]
) -> None:
    """Replace the README generated-results block using real aggregate metrics."""
    path = Path(readme_path)
    text = path.read_text(encoding="utf-8")
    start_marker = "<!-- RESULTS:START -->"
    end_marker = "<!-- RESULTS:END -->"
    if start_marker not in text or end_marker not in text:
        raise ValueError("README is missing generated-results markers")
    header = "| Model | Accuracy | Valid JSON | Schema | Calc validity | Consistency | Avg reward |\n|---|---:|---:|---:|---:|---:|---:|"
    lines = [header]
    for model_name, values in metrics.items():
        lines.append(
            "| {name} | {final_answer_accuracy:.1f}% | {valid_json_percentage:.1f}% | "
            "{schema_compliance_percentage:.1f}% | {calculation_validity_rate:.1f}% | "
            "{final_consistency_percentage:.1f}% | {average_total_reward:.3f} |".format(
                name=model_name, **values
            )
        )
    replacement = start_marker + "\n" + "\n".join(lines) + "\n" + end_marker
    before, remainder = text.split(start_marker, 1)
    _, after = remainder.split(end_marker, 1)
    path.write_text(before + replacement + after, encoding="utf-8")


def evaluate(config: Mapping[str, Any], *, update_readme: bool = False) -> Path:
    """Generate on the same held-out examples with base and aligned policies."""
    model_config = require_mapping(dict(config), "model")
    evaluation = require_mapping(dict(config), "evaluation")
    model_id = str(model_config.get("id", "Qwen/Qwen2.5-0.5B-Instruct"))
    adapter_path = Path(str(evaluation.get("adapter_path", "outputs/reasonforge-adapter")))
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Trained adapter not found at {adapter_path}; train ReasonForge before comparison"
        )
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
    # Sequential loading keeps peak memory reasonable on one T4. Identical test
    # rows and settings make the comparison paired and reproducible.
    for model_name, model_adapter in (("base", None), ("aligned", adapter_path)):
        runner = LazyModelRunner(model_id, model_adapter)
        LOGGER.info("Evaluating %s model on %d held-out examples", model_name, len(dataset))
        for item in dataset:
            response = runner.generate(str(item["question"]), settings)
            rows.append(
                _row(
                    model_name=model_name,
                    item=item,
                    response=response,
                    reward_weights=reward_weights,
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
    metrics = _write_artifacts(rows, output_dir)
    if update_readme:
        update_readme_results("README.md", metrics)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml", help="Evaluation YAML path")
    parser.add_argument(
        "--update-readme", action="store_true", help="Insert these real metrics into README markers"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run held-out evaluation from the command line."""
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
