"""Supervised LoRA warm-start training for ReasonForge structured outputs."""

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
from reasonforge.sft_dataset import prepare_sft_datasets
from reasonforge.training_diagnostics import TrainingHealthCallback

LOGGER = logging.getLogger(__name__)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_sft_training(config: Mapping[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    """Validate and resolve the SFT settings used by both CLI and tests."""
    section = config.get("sft")
    if not isinstance(section, Mapping):
        raise ConfigurationError("Configuration key 'sft' must be a mapping")
    resolved = dict(section)
    if smoke:
        override = config.get("smoke", {})
        if not isinstance(override, Mapping):
            raise ConfigurationError("Configuration key 'smoke' must be a mapping")
        resolved.update(override)
    for key in ("train_size", "validation_size", "max_steps", "max_length"):
        if int(resolved.get(key, 0)) < 1:
            raise ConfigurationError(f"sft.{key} must be a positive integer")
    if float(resolved.get("max_grad_norm", 0.0)) <= 0:
        raise ConfigurationError("sft.max_grad_norm must be positive")
    return resolved


def train_sft(
    config: Mapping[str, Any], *, smoke: bool = False, resume: str | bool | None = None
) -> Path:
    """Train and save the independently loadable SFT LoRA adapter."""
    try:
        import torch
        from peft import LoraConfig
        from torch.utils.data import SequentialSampler
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install requirements-colab.txt before SFT training") from exc

    model_config = require_mapping(dict(config), "model")
    lora_config = require_mapping(dict(config), "lora")
    settings = resolve_sft_training(config, smoke=smoke)
    prepared_config = dict(config)
    prepared_config["sft"] = settings
    prepared = prepare_sft_datasets(prepared_config)
    model_id = str(model_config.get("id", "Qwen/Qwen2.5-0.5B-Instruct"))
    output_dir = Path(str(settings.get("output_dir", "outputs/sft-adapter")))
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class CurriculumSFTTrainer(SFTTrainer):
        def _get_train_sampler(self, train_dataset: Any = None) -> Any:
            selected = self.train_dataset if train_dataset is None else train_dataset
            return SequentialSampler(selected)

    use_fp16 = bool(torch.cuda.is_available())
    optimizer = str(settings.get("optimizer", "adamw_torch"))
    args = SFTConfig(
        output_dir=str(output_dir),
        run_name=str(settings.get("run_name", "reasonforge-sft")),
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
        per_device_train_batch_size=int(settings.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(settings.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(settings.get("gradient_accumulation_steps", 4)),
        learning_rate=float(settings.get("learning_rate", 2.0e-5)),
        max_steps=int(settings.get("max_steps", 100)),
        warmup_ratio=float(settings.get("warmup_ratio", 0.05)),
        max_grad_norm=float(settings.get("max_grad_norm", 0.5)),
        logging_steps=int(settings.get("logging_steps", 1)),
        logging_first_step=True,
        logging_nan_inf_filter=False,
        eval_strategy="steps",
        eval_steps=int(settings.get("eval_steps", 25)),
        save_strategy="steps",
        save_steps=int(settings.get("save_steps", 25)),
        save_total_limit=int(settings.get("save_total_limit", 2)),
        gradient_checkpointing=bool(settings.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=use_fp16,
        bf16=False,
        optim=optimizer,
        report_to=str(settings.get("report_to", "none")),
        max_length=int(settings.get("max_length", 512)),
        completion_only_loss=True,
        packing=False,
        remove_unused_columns=True,
        model_init_kwargs={
            "torch_dtype": "float16" if use_fp16 else "float32",
            "use_cache": False,
        },
    )
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
    health = TrainingHealthCallback(output_dir / "training_health.json")
    trainer = CurriculumSFTTrainer(
        model=model_id,
        args=args,
        train_dataset=prepared.train,
        eval_dataset=prepared.validation,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[health],
    )
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    health.write()
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(prepared.manifest, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "stage": "sft",
        "model_id": model_id,
        "output_dir": str(output_dir),
        "seed": int(config.get("seed", 42)),
        "smoke": smoke,
        "train_metrics": dict(result.metrics),
        "dataset_fingerprint_sha256": prepared.manifest["fingerprint_sha256"],
        "python": platform.python_version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {
            name: _package_version(name)
            for name in ("torch", "transformers", "datasets", "trl", "peft", "sympy")
        },
        "resolved_sft": settings,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(prepared_config, sort_keys=False), encoding="utf-8"
    )
    LOGGER.info("Saved SFT adapter, manifest, and diagnostics to %s", output_dir)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.yaml", help="SFT YAML path")
    parser.add_argument("--smoke", action="store_true", help="Use the short smoke-test override")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help="Resume the latest checkpoint or a supplied checkpoint path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        train_sft(load_yaml(args.config), smoke=args.smoke, resume=args.resume)
    except (ConfigurationError, RuntimeError, ValueError, OSError) as exc:
        LOGGER.error("SFT training failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
