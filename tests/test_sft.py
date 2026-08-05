import pytest

from reasonforge.config import ConfigurationError
from reasonforge.sft import resolve_sft_training
from reasonforge.training_diagnostics import TrainingHealthCallback


def test_sft_smoke_overrides_are_explicit() -> None:
    config = {
        "sft": {
            "train_size": 100,
            "validation_size": 20,
            "max_steps": 50,
            "max_length": 512,
            "max_grad_norm": 0.5,
        },
        "smoke": {"train_size": 8, "validation_size": 2, "max_steps": 2},
    }
    assert resolve_sft_training(config, smoke=True)["train_size"] == 8
    assert resolve_sft_training(config, smoke=False)["max_steps"] == 50


def test_sft_config_requires_explicit_stability_bounds() -> None:
    with pytest.raises(ConfigurationError):
        resolve_sft_training(
            {
                "sft": {
                    "train_size": 1,
                    "validation_size": 1,
                    "max_steps": 1,
                    "max_length": 32,
                    "max_grad_norm": 0,
                }
            },
        )


def test_health_callback_preserves_nonfinite_evidence(tmp_path) -> None:
    callback = TrainingHealthCallback(tmp_path / "health.json")
    state = type("State", (), {"global_step": 3})()
    callback.on_log(None, state, None, {"loss": float("nan"), "grad_norm": 1.25})
    summary = callback.summary()
    assert summary["nonfinite_event_count"] == 1
    assert summary["finite_metric_ranges"]["grad_norm"]["maximum"] == 1.25
