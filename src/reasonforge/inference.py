"""Lazy model loading and deterministic text generation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reasonforge.dataset import SYSTEM_PROMPT

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationSettings:
    """Basic decoding controls shared by evaluation and the Gradio app."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.95
    seed: int = 42


@dataclass(frozen=True)
class GenerationResult:
    """Generated text with exact token-level stopping metadata."""

    text: str
    completion_tokens: int
    truncated: bool
    finish_reason: str


def generation_finish_metadata(
    token_ids: list[int], *, max_new_tokens: int, eos_token_ids: set[int]
) -> tuple[bool, str]:
    """Classify an observed token sequence without decoding heuristics."""
    ended_with_eos = bool(token_ids and token_ids[-1] in eos_token_ids)
    truncated = len(token_ids) >= max_new_tokens and not ended_with_eos
    if ended_with_eos:
        return False, "eos_token"
    if truncated:
        return True, "max_new_tokens"
    return False, "generation_stopped"


class LazyModelRunner:
    """Load a base model (and optional LoRA adapter) only on first generation."""

    def __init__(self, model_id: str, adapter_path: str | Path | None = None) -> None:
        self.model_id = model_id
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None

    @property
    def adapter_available(self) -> bool:
        """Whether the configured adapter path looks loadable."""
        return (
            self.adapter_path is not None and (self.adapter_path / "adapter_config.json").is_file()
        )

    def load(self) -> None:
        """Load tokenizer/model weights and attach the adapter when configured."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install ReasonForge model dependencies before generation") from exc
        if self.adapter_path is not None and not self.adapter_available:
            raise FileNotFoundError(
                f"No LoRA adapter found at {self.adapter_path}. Run training first or choose another path."
            )
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        LOGGER.info("Loading %s on %s", self.model_id, "CUDA" if use_cuda else "CPU")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            low_cpu_mem_usage=True,
        )
        if self.adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("PEFT is required to load a ReasonForge adapter") from exc
            model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        if not use_cuda:
            model.to("cpu")
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    def generate(self, problem: str, settings: GenerationSettings) -> str:
        """Generate the assistant continuation for one math problem."""
        return self.generate_result(problem, settings).text

    def generate_result(self, problem: str, settings: GenerationSettings) -> GenerationResult:
        """Generate one response and retain exact finish/truncation evidence."""
        if not problem.strip():
            raise ValueError("Problem cannot be empty")
        if not 1 <= settings.max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be between 1 and 1024")
        if settings.temperature < 0:
            raise ValueError("temperature cannot be negative")
        self.load()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        torch.manual_seed(settings.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(settings.seed)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.strip()},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt")
        target_device = next(model.parameters()).device
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        do_sample = settings.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=settings.temperature, top_p=settings.top_p)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        token_ids = [int(value) for value in new_tokens.tolist()]
        configured_eos = tokenizer.eos_token_id
        if isinstance(configured_eos, int):
            eos_ids = {configured_eos}
        elif configured_eos is None:
            eos_ids = set()
        else:
            eos_ids = {int(value) for value in configured_eos}
        truncated, finish_reason = generation_finish_metadata(
            token_ids,
            max_new_tokens=settings.max_new_tokens,
            eos_token_ids=eos_ids,
        )
        return GenerationResult(
            text=str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip(),
            completion_tokens=len(token_ids),
            truncated=truncated,
            finish_reason=finish_reason,
        )
