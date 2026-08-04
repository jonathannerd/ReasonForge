"""Lazy Gradio interface for side-by-side ReasonForge inspection."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reasonforge.inference import GenerationSettings, LazyModelRunner
from reasonforge.parsing import parse_completion
from reasonforge.rewards import score_completion
from reasonforge.verifier import verify_completion

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _inspect(response: str, reference_answer: str) -> tuple[Any, Any, Any]:
    parsed = parse_completion(response)
    parsed_json = parsed.data if parsed.data is not None else {"parse_error": parsed.error}
    if reference_answer.strip():
        verification = verify_completion(response, reference_answer.strip()).model_dump(
            mode="json", exclude={"parse"}
        )
        rewards = score_completion(response, reference_answer.strip()).model_dump(
            mode="json", by_alias=True
        )
    else:
        if parsed.solution is not None:
            verification = verify_completion(response, parsed.solution.final_answer).model_dump(
                mode="json", exclude={"parse"}
            )
            verification.update(
                answer_correct=None,
                normalized_reference=None,
                note="No external reference supplied; answer correctness is unscored. Calculation checks remain active.",
            )
        else:
            verification = {
                "note": "Enter a reference answer to score correctness; no schema-valid calculations were available.",
                "schema_valid": parsed.schema_valid,
            }
        rewards = {"note": "Reference answer required for reward scoring."}
    return parsed_json, verification, rewards


def build_app(
    model_id: str = DEFAULT_MODEL_ID, adapter_path: str | Path = "outputs/reasonforge-adapter"
) -> Any:
    """Construct the interface without loading model weights."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install Gradio to launch the ReasonForge application") from exc

    base = LazyModelRunner(model_id)
    aligned = LazyModelRunner(model_id, adapter_path)

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
        try:
            base_response = base.generate(problem, settings)
            base_details = _inspect(base_response, reference)
        except Exception as exc:  # UI boundary: surface actionable errors, never crash the app.
            base_response = f"Base generation unavailable: {exc}"
            base_details = ({"error": str(exc)}, {"error": str(exc)}, {"error": str(exc)})
        if aligned.adapter_available:
            try:
                aligned_response = aligned.generate(problem, settings)
                aligned_details = _inspect(aligned_response, reference)
            except Exception as exc:
                aligned_response = f"Aligned generation unavailable: {exc}"
                aligned_details = ({"error": str(exc)}, {"error": str(exc)}, {"error": str(exc)})
        else:
            message = (
                f"No trained adapter found at {Path(adapter_path)}. Run `python -m reasonforge.train "
                "--config configs/training.yaml` on a GPU, then relaunch this app."
            )
            aligned_response = message
            aligned_details = ({"status": message}, {"status": message}, {"status": message})
        return (
            base_response,
            *base_details,
            aligned_response,
            *aligned_details,
        )

    with gr.Blocks(title="ReasonForge") as demo:
        gr.Markdown(
            "# ReasonForge\nCompare an untouched Qwen model with its GRPO + LoRA adapter. "
            "Verification covers only the displayed structured arithmetic—not hidden reasoning."
        )
        problem = gr.Textbox(
            label="Math problem",
            lines=3,
            value="A box has 12 rows of 5 pencils. How many pencils are there?",
        )
        reference = gr.Textbox(
            label="Reference answer (optional but required for correctness rewards)", value="60"
        )
        with gr.Row():
            max_tokens = gr.Slider(32, 512, value=256, step=16, label="Maximum new tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            seed = gr.Number(value=42, precision=0, label="Seed")
        run = gr.Button("Compare models", variant="primary")
        outputs: list[Any] = []
        with gr.Row():
            for title in ("Base model", "Aligned model"):
                with gr.Column():
                    gr.Markdown(f"## {title}")
                    outputs.extend(
                        [
                            gr.Textbox(label="Raw response", lines=10),
                            gr.JSON(label="Parsed JSON"),
                            gr.JSON(label="Verification and calculation checks"),
                            gr.JSON(label="Reward components"),
                        ]
                    )
        run.click(
            compare,
            inputs=[problem, reference, max_tokens, temperature, top_p, seed],
            outputs=outputs,
        )
    return demo


def build_parser() -> argparse.ArgumentParser:
    """Build the Gradio CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face base model ID")
    parser.add_argument(
        "--adapter-path", default="outputs/reasonforge-adapter", help="LoRA adapter directory"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a temporary public Gradio link"
    )
    parser.add_argument("--port", type=int, default=7860, help="Local server port")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the app explicitly; importing this module has no model/UI side effects."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        app = build_app(args.model_id, args.adapter_path)
        app.launch(share=args.share, server_port=args.port)
    except (RuntimeError, OSError, ValueError) as exc:
        LOGGER.error("Application failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
