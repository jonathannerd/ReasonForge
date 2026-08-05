"""Finite-value and gradient diagnostics shared by SFT and GRPO runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class TrainingHealthCallback:
    """Collect Trainer logs without filtering NaN/Inf evidence."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.logs: list[dict[str, Any]] = []
        self.nonfinite_events: list[dict[str, Any]] = []

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> None:
        if not isinstance(logs, Mapping):
            return
        record = {"step": int(getattr(state, "global_step", 0)), **dict(logs)}
        self.logs.append(record)
        for key in ("loss", "eval_loss", "grad_norm", "learning_rate"):
            value = record.get(key)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                self.nonfinite_events.append(
                    {"step": record["step"], "metric": key, "value": str(value)}
                )

    def summary(self) -> dict[str, Any]:
        numeric: dict[str, list[float]] = {}
        for record in self.logs:
            for key in ("loss", "eval_loss", "grad_norm", "learning_rate"):
                value = record.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    numeric.setdefault(key, []).append(float(value))
        return {
            "logged_steps": len(self.logs),
            "nonfinite_event_count": len(self.nonfinite_events),
            "nonfinite_events": self.nonfinite_events,
            "finite_metric_ranges": {
                key: {"minimum": min(values), "maximum": max(values), "last": values[-1]}
                for key, values in sorted(numeric.items())
            },
            "optimizer_overflow_skips_detected": None,
            "optimizer_overflow_note": (
                "The pinned Trainer stack does not expose AMP overflow skips in callback logs; "
                "this field is null rather than inferred."
            ),
        }

    def write(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(self.summary(), indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> None:
        self.write()
