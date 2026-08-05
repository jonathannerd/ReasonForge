"""Auditable, TRL-compatible reward components for structured mathematics."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from statistics import mean, pvariance
from typing import Any

from reasonforge.parsing import completion_to_text
from reasonforge.schemas import RewardBreakdown
from reasonforge.verifier import likely_truncated_text, verify_completion

DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "schema": 1.0,
    "answer_correctness": 5.0,
    "calculation_validity": 2.0,
    "final_consistency": 1.5,
    "conciseness": 0.5,
    "suspicious_penalty": -1.0,
}

_SUSPICIOUS_RE = re.compile(
    r"(?:__|import\b|eval\s*\(|exec\s*\(|subprocess|os\.|system\s*\(|<script)",
    re.IGNORECASE,
)


class RewardDiagnosticsCollector:
    """Aggregate the group-level learning signal GRPO actually observes."""

    def __init__(self, num_generations: int) -> None:
        if num_generations < 2:
            raise ValueError("Reward diagnostics require at least two generations")
        self.num_generations = num_generations
        self.groups: list[dict[str, Any]] = []

    def record(
        self,
        completions: Sequence[object],
        references: Sequence[str],
        scores: Sequence[RewardBreakdown],
    ) -> None:
        if len(completions) % self.num_generations:
            raise ValueError("Completion count must be divisible by num_generations")
        for start in range(0, len(completions), self.num_generations):
            end = start + self.num_generations
            group_scores = scores[start:end]
            correctness = [float(score.answer_correctness > 0) for score in group_scores]
            totals = [float(score.total) for score in group_scores]
            structure = [
                float(score.schema_score + score.calculation_validity + score.final_consistency)
                for score in group_scores
            ]
            texts = [completion_to_text(value) for value in completions[start:end]]
            self.groups.append(
                {
                    "reference_answer": references[start],
                    "correct_completions": int(sum(correctness)),
                    "all_incorrect": not any(correctness),
                    "all_correct": all(correctness),
                    "correctness_variance": pvariance(correctness),
                    "total_reward_variance": pvariance(totals),
                    "mean_total_reward": mean(totals),
                    "mean_structure_reward": mean(structure),
                    "unique_generations": len(set(texts)),
                    "likely_truncated_completions": sum(
                        likely_truncated_text(text) for text in texts
                    ),
                }
            )

    def summary(self) -> dict[str, Any]:
        count = len(self.groups)
        if not count:
            return {
                "groups": 0,
                "num_generations": self.num_generations,
                "note": "No reward groups were observed.",
            }

        def group_count(key: str) -> int:
            return sum(bool(group[key]) for group in self.groups)

        return {
            "groups": count,
            "num_generations": self.num_generations,
            "groups_with_any_correct": count - group_count("all_incorrect"),
            "all_incorrect_groups": group_count("all_incorrect"),
            "all_correct_groups": group_count("all_correct"),
            "zero_correctness_variance_groups": sum(
                float(group["correctness_variance"]) == 0.0 for group in self.groups
            ),
            "zero_total_reward_variance_groups": sum(
                float(group["total_reward_variance"]) == 0.0 for group in self.groups
            ),
            "mean_correctness_variance": mean(
                float(group["correctness_variance"]) for group in self.groups
            ),
            "mean_total_reward_variance": mean(
                float(group["total_reward_variance"]) for group in self.groups
            ),
            "mean_structure_reward": mean(
                float(group["mean_structure_reward"]) for group in self.groups
            ),
            "mean_unique_generations": mean(
                int(group["unique_generations"]) for group in self.groups
            ),
            "likely_truncated_completions": sum(
                int(group["likely_truncated_completions"]) for group in self.groups
            ),
            "per_group": self.groups,
        }


def _finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def _reference_list(kwargs: Mapping[str, Any], expected: int) -> list[str]:
    candidates = kwargs.get("reference_answer", kwargs.get("ground_truth"))
    if isinstance(candidates, str):
        values = [candidates]
    elif isinstance(candidates, Sequence):
        values = [str(item) for item in candidates]
    else:
        values = []
    if values and len(values) < expected and expected % len(values) == 0:
        generations_per_prompt = expected // len(values)
        values = [value for value in values for _ in range(generations_per_prompt)]
    if len(values) != expected:
        raise ValueError(
            "Reward functions require one reference_answer per completion "
            f"(received {len(values)} for {expected})"
        )
    return values


def score_completion(
    completion: object,
    reference_answer: str,
    weights: Mapping[str, float] | None = None,
    *,
    concise_char_limit: int = 600,
) -> RewardBreakdown:
    """Compute all bounded reward components for one untrusted completion.

    Correctness is worth five times full schema compliance. Calculation rewards
    cover only supported structured arithmetic, not free-form or hidden reasoning.
    """
    configured = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}
    verification = verify_completion(completion, reference_answer)
    parsed = verification.parse
    text = completion_to_text(completion)

    schema_fraction = (
        1.0 if parsed.schema_valid and parsed.exact_json else 0.25 if parsed.json_valid else 0.0
    )
    schema = configured["schema"] * schema_fraction
    answer = configured["answer_correctness"] if verification.answer_correct else 0.0
    calculations = configured["calculation_validity"] * verification.calculation_validity_rate
    consistency = configured["final_consistency"] if verification.final_consistent else 0.0
    concise = (
        configured["conciseness"]
        if verification.answer_correct and 0 < len(text) <= concise_char_limit
        else 0.0
    )
    suspicious = configured["suspicious_penalty"] if _SUSPICIOUS_RE.search(text) else 0.0
    components = [schema, answer, calculations, consistency, concise, suspicious]
    components = [_finite(value) for value in components]
    total = _finite(sum(components))
    return RewardBreakdown(
        schema_score=components[0],
        answer_correctness=components[1],
        calculation_validity=components[2],
        final_consistency=components[3],
        conciseness=components[4],
        suspicious_penalty=components[5],
        total=total,
    )


def _component_reward(
    component: str,
    weights: Mapping[str, float] | None = None,
    scorer: Callable[[object, str], RewardBreakdown] | None = None,
    observer: Callable[[Sequence[object], Sequence[str], Sequence[RewardBreakdown]], None]
    | None = None,
) -> Callable[..., list[float]]:
    """Create a named reward callable so TRL logs each component separately."""

    def reward(completions: Sequence[object], **kwargs: Any) -> list[float]:
        references = _reference_list(kwargs, len(completions))
        attribute = "schema_score" if component == "schema" else component
        score = scorer or (
            lambda completion, reference: score_completion(completion, reference, weights)
        )
        scored = [
            score(completion, reference)
            for completion, reference in zip(completions, references, strict=True)
        ]
        if observer is not None:
            observer(completions, references, scored)
        values = [getattr(value, attribute) for value in scored]
        return [_finite(value) for value in values]

    reward.__name__ = f"{component}_reward"
    reward.__doc__ = {
        "schema": "Rewards exact JSON and strict schema compliance; JSON alone earns only partial credit.",
        "answer_correctness": "Strongly rewards mathematical equivalence to the reference answer.",
        "calculation_validity": "Rewards the fraction of restricted structured calculations that verify.",
        "final_consistency": "Rewards agreement between the last verified calculation and final answer.",
        "conciseness": "Rewards bounded response length only after the answer is mathematically correct.",
        "suspicious_penalty": "Penalizes obvious code-execution or injection-like tokens in output.",
    }[component]
    return reward


def build_reward_functions(
    weights: Mapping[str, float] | None = None,
    diagnostics: RewardDiagnosticsCollector | None = None,
) -> list[Callable[..., list[float]]]:
    """Return separately named reward functions accepted by TRL ``GRPOTrainer``."""

    @lru_cache(maxsize=8_192)
    def cached_score(completion_text: str, reference_answer: str) -> RewardBreakdown:
        return score_completion(completion_text, reference_answer, weights)

    def scorer(completion: object, reference_answer: str) -> RewardBreakdown:
        return cached_score(completion_to_text(completion), reference_answer)

    return [
        _component_reward(
            component,
            weights,
            scorer,
            diagnostics.record if diagnostics is not None and component == "schema" else None,
        )
        for component in (
            "schema",
            "answer_correctness",
            "calculation_validity",
            "final_consistency",
            "conciseness",
            "suspicious_penalty",
        )
    ]


# Convenient default-weight functions for direct use and tests.
schema_reward = _component_reward("schema")
answer_correctness_reward = _component_reward("answer_correctness")
calculation_validity_reward = _component_reward("calculation_validity")
final_consistency_reward = _component_reward("final_consistency")
conciseness_reward = _component_reward("conciseness")
suspicious_penalty_reward = _component_reward("suspicious_penalty")
