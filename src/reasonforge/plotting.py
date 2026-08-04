"""Comparison plots for real (never fabricated) evaluation artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def plot_comparison(metrics: Mapping[str, Mapping[str, Any]], output_path: str | Path) -> Path:
    """Save a two-panel aggregate-metric and failure-category comparison."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError("Install plotting dependencies to generate evaluation figures") from exc

    percentage_metrics = [
        "final_answer_accuracy",
        "valid_json_percentage",
        "schema_compliance_percentage",
        "calculation_validity_rate",
        "final_consistency_percentage",
    ]
    rows = [
        {"model": model_name, "metric": metric.replace("_", " "), "value": values[metric]}
        for model_name, values in metrics.items()
        for metric in percentage_metrics
    ]
    failure_rows = [
        {"model": model_name, "category": category.replace("_", " "), "count": count}
        for model_name, values in metrics.items()
        for category, count in values.get("failure_categories", {}).items()
    ]
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=pd.DataFrame(rows), x="value", y="metric", hue="model", ax=axes[0])
    axes[0].set(
        xlabel="Rate (%)", ylabel="", title="Base vs. aligned verification metrics", xlim=(0, 100)
    )
    if failure_rows:
        sns.barplot(
            data=pd.DataFrame(failure_rows),
            x="count",
            y="category",
            hue="model",
            ax=axes[1],
        )
    axes[1].set(xlabel="Examples", ylabel="", title="Failure categories")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
