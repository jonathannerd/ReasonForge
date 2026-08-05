"""Comparison plots for real (never fabricated) evaluation artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _plotting() -> tuple[Any, Any, Any]:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError("Install plotting dependencies to generate evaluation figures") from exc
    sns.set_theme(style="whitegrid")
    return plt, pd, sns


def plot_comparison(metrics: Mapping[str, Mapping[str, Any]], output_path: str | Path) -> Path:
    """Save core math, format, consistency, and truncation comparisons."""
    plt, pd, sns = _plotting()
    percentage_metrics = [
        "math_accuracy",
        "strict_end_to_end_accuracy",
        "json_validity_percentage",
        "schema_compliance_percentage",
        "calculation_validity_rate",
        "final_consistency_percentage",
        "truncation_rate",
    ]
    rows = [
        {"model": model_name, "metric": metric.replace("_", " "), "value": values[metric]}
        for model_name, values in metrics.items()
        for metric in percentage_metrics
    ]
    figure, axis = plt.subplots(figsize=(12, 7))
    sns.barplot(data=pd.DataFrame(rows), x="value", y="metric", hue="model", ax=axis)
    axis.set(xlabel="Rate (%)", ylabel="", title="Held-out ReasonForge evaluation", xlim=(0, 100))
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_failures(metrics: Mapping[str, Mapping[str, Any]], output_path: str | Path) -> Path:
    """Save primary failure categories across all evaluated models."""
    plt, pd, sns = _plotting()
    rows = [
        {"model": model_name, "category": category.replace("_", " "), "count": count}
        for model_name, values in metrics.items()
        for category, count in values.get("failure_categories", {}).items()
    ]
    figure, axis = plt.subplots(figsize=(12, 7))
    if rows:
        sns.barplot(data=pd.DataFrame(rows), x="count", y="category", hue="model", ax=axis)
    axis.set(xlabel="Examples", ylabel="", title="Primary failure categories")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_reward_components(
    metrics: Mapping[str, Mapping[str, Any]], output_path: str | Path
) -> Path:
    """Save mean reward components rather than only the aggregate reward."""
    plt, pd, sns = _plotting()
    component_keys = [
        "average_reward_schema",
        "average_reward_answer_correctness",
        "average_reward_calculation_validity",
        "average_reward_final_consistency",
        "average_reward_conciseness",
        "average_reward_suspicious_penalty",
    ]
    rows = [
        {
            "model": model_name,
            "component": key.removeprefix("average_reward_").replace("_", " "),
            "value": values[key],
        }
        for model_name, values in metrics.items()
        for key in component_keys
    ]
    figure, axis = plt.subplots(figsize=(12, 7))
    sns.barplot(data=pd.DataFrame(rows), x="value", y="component", hue="model", ax=axis)
    axis.set(xlabel="Mean weighted reward", ylabel="", title="Reward components")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
