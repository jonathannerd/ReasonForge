"""Explicit GRPO + LoRA training entry point for ReasonForge."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from reasonforge.config import ConfigurationError, load_yaml, require_mapping
from reasonforge.dataset import prepare_datasets
from reasonforge.rewards import build_reward_functions

LOGGER = logging.getLogger(__name__)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_batch(training: Mapping[str, Any]) -> None:
    generations = int(training.get("num_generations", 4))
    batch = int(training.get("per_device_train_batch_size", 1))
    accumulation = int(training.get("gradient_accumulation_steps", 4))
    if generations < 2:
        raise ConfigurationError("GRPO requires at least two generations per prompt")
    if (batch * accumulation) % generations:
        raise ConfigurationError(
            "per_device_train_batch_size * gradient_accumulation_steps must be "
            "divisible by num_generations on a single process"
        )


def _optimizer_name(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb

            has_optimizer = hasattr(bnb, "optim") and hasattr(bnb.optim, "AdamW8bit")
        except Exception as exc:
            LOGGER.warning("bitsandbytes is installed but unusable (%s); falling back", exc)
        else:
            if has_optimizer:
                LOGGER.info("Using bitsandbytes paged AdamW 8-bit optimizer")
                return "paged_adamw_8bit"
    LOGGER.info("bitsandbytes/CUDA unavailable; using torch AdamW fallback")
    return "adamw_torch"


def train(
    config: Mapping[str, Any], *, fallback: bool = False, resume: str | bool | None = None
) -> Path:
    """Run GRPO training and save a LoRA adapter plus reproducibility metadata."""
    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install requirements-colab.txt first."
        ) from exc

    model_config = require_mapping(dict(config), "model")
    training_config = dict(require_mapping(dict(config), "training"))
    lora_config = require_mapping(dict(config), "lora")
    if fallback:
        fallback_config = config.get("fallback", {})
        if isinstance(fallback_config, Mapping):
            training_config.update(fallback_config)
    _validate_batch(training_config)

    model_id = str(model_config.get("id", "Qwen/Qwen2.5-0.5B-Instruct"))
    output_dir = Path(str(training_config.get("output_dir", "outputs/reasonforge-adapter")))
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = prepare_datasets(config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_fp16 = bool(torch.cuda.is_available())
    optimizer = _optimizer_name(torch, str(training_config.get("optimizer", "auto")))
    model_init_kwargs: dict[str, Any] = {
        "torch_dtype": "float16" if use_fp16 else "float32",
        "use_cache": False,
    }
    args = GRPOConfig(
        output_dir=str(output_dir),
        run_name=str(training_config.get("run_name", "reasonforge-grpo")),
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
        per_device_train_batch_size=int(training_config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_config.get("gradient_accumulation_steps", 4)),
        num_generations=int(training_config.get("num_generations", 4)),
        max_prompt_length=int(training_config.get("max_prompt_length", 512)),
        max_completion_length=int(training_config.get("max_completion_length", 256)),
        learning_rate=float(training_config.get("learning_rate", 1.0e-5)),
        num_train_epochs=float(training_config.get("num_train_epochs", 1)),
        max_steps=int(training_config.get("max_steps", -1)),
        warmup_ratio=float(training_config.get("warmup_ratio", 0.05)),
        logging_steps=int(training_config.get("logging_steps", 1)),
        save_steps=int(training_config.get("save_steps", 25)),
        save_total_limit=int(training_config.get("save_total_limit", 2)),
        gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=use_fp16,
        bf16=False,
        optim=optimizer,
        temperature=float(training_config.get("temperature", 0.8)),
        top_p=float(training_config.get("top_p", 0.95)),
        beta=float(training_config.get("beta", 0.04)),
        loss_type=str(training_config.get("loss_type", "dapo")),
        remove_unused_columns=False,
        report_to=str(training_config.get("report_to", "none")),
        log_completions=bool(training_config.get("log_completions", False)),
        model_init_kwargs=model_init_kwargs,
    )
    # Qwen2.5 names these attention/MLP projections consistently. Targeting all
    # seven projections gives a compact QLoRA-style adapter without touching
    # embeddings or lm_head.
    peft_config = LoraConfig(
        r=int(lora_config.get("rank", 16)),
        lora_alpha=int(lora_config.get("alpha", 32)),
        lora_dropout=float(lora_config.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(
            lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
    )
    reward_weights = config.get("rewards")
    if not isinstance(reward_weights, Mapping):
        reward_weights = None
    trainer = GRPOTrainer(
        model=model_id,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        reward_funcs=build_reward_functions(reward_weights),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": model_id,
        "seed": int(config.get("seed", 42)),
        "fallback": fallback,
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {
            name: _package_version(name)
            for name in ("torch", "transformers", "datasets", "trl", "peft", "sympy")
        },
        "config": config,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    LOGGER.info("Saved ReasonForge adapter and metadata to %s", output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml", help="Training YAML path")
    parser.add_argument(
        "--fallback", action="store_true", help="Use the two-generation low-memory override"
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume the latest checkpoint or a supplied checkpoint path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI wrapper that converts configuration/runtime failures into useful errors."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_yaml(args.config)
        train(config, fallback=args.fallback, resume=args.resume)
    except (ConfigurationError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.error("Training failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
