"""Update README's generated-results block from an existing metrics artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reasonforge.evaluate import update_readme_results


def main() -> int:
    """Load measured aggregate metrics and update the marked README section."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics", default="results/latest/aggregate_metrics.json", help="Metrics JSON path"
    )
    parser.add_argument("--readme", default="README.md", help="README path")
    args = parser.parse_args()
    metrics_path = Path(args.metrics)
    try:
        metrics: Any = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise ValueError("metrics JSON must contain a model-to-metrics mapping")
        update_readme_results(args.readme, metrics)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
