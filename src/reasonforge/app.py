"""Lazy Gradio comparison of base, SFT, and SFT-plus-GRPO policies."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reasonforge.inference import GenerationSettings, LazyModelRunner
from reasonforge.parsing import parse_completion
from reasonforge.rewards import score_completion
from reasonforge.verifier import extract_final_answer, verify_completion

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _inspect(response: str, reference_answer: str) -> tuple[Any, Any, Any]:
    parsed = parse_completion(response)
    parsed_json = parsed.data if parsed.data is not None else {"parse_error": parsed.error}
    extracted = extract_final_answer(response).model_dump(mode="json")
    if reference_answer.strip():
        verification = verify_completion(response, reference_answer.strip()).model_dump(
            mode="json", exclude={"parse"}
        )
        verification["independent_answer_extraction"] = extracted
        rewards = score_completion(response, reference_answer.strip()).model_dump(
            mode="json", by_alias=True
        )
    else:
        verification = {
            "independent_answer_extraction": extracted,
            "schema_valid": parsed.schema_valid,
            "note": "Enter an external reference to score math accuracy and rewards.",
        }
        rewards = {"note": "Reference answer required for reward scoring."}
    return parsed_json, verification, rewards


def build_app(
    model_id: str = DEFAULT_MODEL_ID,
    sft_adapter_path: str | Path = "outputs/sft-adapter",
    grpo_adapter_path: str | Path = "outputs/sft-grpo-adapter",
) -> Any:
    """Construct the interface without loading model weights."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install Gradio to launch the ReasonForge application") from exc

    runners = {
        "Base": LazyModelRunner(model_id),
        "SFT": LazyModelRunner(model_id, sft_adapter_path),
        "SFT + GRPO": LazyModelRunner(model_id, grpo_adapter_path),
    }

    def compare(
        problem: str,
        reference: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> tuple[Any, ...]:
        settings = GenerationSettings(
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            seed=int(seed),
        )
        output: list[Any] = []
        for name, runner in runners.items():
            if name != "Base" and not runner.adapter_available:
                message = f"No adapter found at {runner.adapter_path}. Run the corresponding training stage."
                output.extend(
                    [message, {"status": message}, {"status": message}, {"status": message}, {}]
                )
                continue
            try:
                generated = runner.generate_result(problem, settings)
                parsed, verification, rewards = _inspect(generated.text, reference)
                generation = {
                    "completion_tokens": generated.completion_tokens,
                    "finish_reason": generated.finish_reason,
                    "truncated": generated.truncated,
                }
                output.extend([generated.text, parsed, verification, rewards, generation])
            except Exception as exc:  # UI boundary: return actionable context instead of crashing.
                message = f"{name} generation unavailable: {exc}"
                output.extend(
                    [message, {"error": str(exc)}, {"error": str(exc)}, {"error": str(exc)}, {}]
                )
        return tuple(output)

    with gr.Blocks(title="ReasonForge") as demo:
        gr.Markdown(
            "# ReasonForge\nCompare the base model, structured-output SFT adapter, and its GRPO "
            "continuation. Math extraction is reported separately from strict JSON verification."
        )
        problem = gr.Textbox(
            label="Math problem",
            lines=3,
            value="A box has 12 rows of 5 pencils. How many pencils are there?",
        )
        reference = gr.Textbox(
            label="Reference answer (optional; required for accuracy and rewards)", value="60"
        )
        with gr.Row():
            max_tokens = gr.Slider(32, 512, value=256, step=16, label="Maximum new tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            seed = gr.Number(value=42, precision=0, label="Seed")
        run = gr.Button("Compare three models", variant="primary")
        outputs: list[Any] = []
        with gr.Row():
            for title in runners:
                with gr.Column():
                    gr.Markdown(f"## {title}")
                    outputs.extend(
                        [
                            gr.Textbox(label="Raw response", lines=10),
                            gr.JSON(label="Parsed JSON"),
                            gr.JSON(label="Math extraction and verification"),
                            gr.JSON(label="Reward components"),
                            gr.JSON(label="Generation metadata"),
                        ]
                    )
        run.click(
            compare,
            inputs=[problem, reference, max_tokens, temperature, top_p, seed],
            outputs=outputs,
        )
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face base model ID")
    parser.add_argument("--sft-adapter", default="outputs/sft-adapter")
    parser.add_argument("--grpo-adapter", default="outputs/sft-grpo-adapter")
    parser.add_argument("--share", action="store_true", help="Create a temporary Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Local server port")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        app = build_app(args.model_id, args.sft_adapter, args.grpo_adapter)
        app.launch(share=args.share, server_port=args.port)
    except (RuntimeError, OSError, ValueError) as exc:
        LOGGER.error("Application failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
