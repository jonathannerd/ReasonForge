"""Build the checked-in ReasonForge Colab tutorial with nbformat."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def markdown(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    markdown(
        """
# ReasonForge v2 — SFT → GRPO on a Colab T4

This reproducible tutorial warms up `Qwen/Qwen2.5-0.5B-Instruct` with structured-output
LoRA SFT, continues the **same adapter** with GRPO, and compares base, SFT, and SFT+GRPO
on paired GSM8K test examples. Expensive cells are labeled. The notebook never treats a
smoke run as a final experiment and stops before meaningful GRPO if its signal or numeric
health diagnostics are unsafe.
"""
    ),
    markdown(
        """
## 1. Check the runtime and record the GPU

Choose **Runtime → Change runtime type → T4 GPU**. This cell is read-only and must show CUDA.
"""
    ),
    code(
        """
import json, os, platform, shutil, subprocess, sys, time
from pathlib import Path
import torch

print("Python:", platform.python_version())
subprocess.run(["nvidia-smi"], check=False)
assert torch.cuda.is_available(), "Select a T4 GPU runtime before training."
print("CUDA device:", torch.cuda.get_device_name(0))
"""
    ),
    markdown(
        """
## 2. Clone the released repository and install the pinned stack

The default URL and branch use the released public ReasonForge repository. Re-running this cell
updates an existing clean checkout instead of nesting it.
"""
    ),
    code(
        """
REPO_URL = "https://github.com/jonathannerd/ReasonForge.git"
BRANCH = "main"
PROJECT_DIR = Path("/content/ReasonForge")

if not PROJECT_DIR.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(PROJECT_DIR)], check=True)
else:
    subprocess.run(["git", "-C", str(PROJECT_DIR), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(PROJECT_DIR), "switch", BRANCH], check=True)
os.chdir(PROJECT_DIR)
%pip install -q -r requirements-colab.txt
# Colab preinstalls vision/audio wheels for a newer Torch release. ReasonForge
# is text-only, so remove those incompatible optional packages before importing
# Transformers/PEFT.
%pip uninstall -y torchvision torchaudio
%pip install -q -e . pytest==8.4.2 ruff==0.13.1
# Colab's already-running kernel does not always reload a newly created
# editable-install .pth file; make this checked-out source tree explicit.
sys.path.insert(0, str(PROJECT_DIR / "src"))
print("Commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
"""
    ),
    markdown(
        """
## 3. Run CPU-safe validation before spending GPU time

This checks the parser, verifier, fallback extraction, SFT targets, reward ordering, adapter
lineage, evaluation metrics, and notebook structure.
"""
    ),
    code(
        """
subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
subprocess.run([sys.executable, "-m", "ruff", "check", "."], check=True)
subprocess.run([sys.executable, "-m", "ruff", "format", "--check", "."], check=True)
"""
    ),
    markdown(
        """
## 4. Audit structured SFT targets and the leakage boundary

Only GSM8K's official **train** split is used for SFT train/validation. Unsupported annotations
are rejected, never repaired with invented reasoning. The manifest records counts, stage mix,
disjointness, rejection reasons, and a SHA-256 dataset fingerprint.
"""
    ),
    code(
        """
import yaml
from reasonforge.sft_dataset import prepare_sft_datasets

sft_config = yaml.safe_load(Path("configs/sft.yaml").read_text())
audit_config = dict(sft_config)
audit_config["sft"] = {**sft_config["sft"], "train_size": 32, "validation_size": 8}
prepared = prepare_sft_datasets(audit_config)
print(json.dumps(prepared.manifest, indent=2))
print(json.dumps(json.loads(prepared.train[0]["target_json"]), indent=2))
assert prepared.manifest["official_test_examples_used"] == 0
assert prepared.manifest["train_validation_overlap"] == 0
"""
    ),
    markdown(
        """
## 5. GPU smoke test: structured-output SFT

**GPU-expensive (short):** four optimizer steps. This validates T4 execution, LoRA saving,
evaluation loss, checkpointing, and the independently loadable SFT artifact.
"""
    ),
    code(
        """
subprocess.run(
    [sys.executable, "-m", "reasonforge.sft", "--config", "configs/sft.yaml", "--smoke"],
    check=True,
)
"""
    ),
    markdown(
        """
## 6. Verify the smoke adapter and numeric health

Do not continue if the adapter is missing, loss/gradient logs contain non-finite values, or
metadata does not identify an SFT stage.
"""
    ),
    code(
        """
sft_smoke = Path("outputs/sft-smoke-adapter")
assert (sft_smoke / "adapter_config.json").is_file()
assert any(sft_smoke.glob("adapter_model.*"))
sft_smoke_meta = json.loads((sft_smoke / "run_metadata.json").read_text())
sft_smoke_health = json.loads((sft_smoke / "training_health.json").read_text())
print(json.dumps({"metadata": sft_smoke_meta, "health": sft_smoke_health}, indent=2))
assert sft_smoke_meta["stage"] == "sft"
assert sft_smoke_meta["precision"] == "fp32"
assert sft_smoke_health["nonfinite_event_count"] == 0
"""
    ),
    markdown(
        """
## 7. GPU meaningful run: SFT LoRA

**GPU-expensive:** the checked-in configuration uses 1,024 validated targets and 100 steps.
It saves to `outputs/sft-adapter`, separate from the smoke artifact. Resume with `--resume` after
an interruption. SFT intentionally uses FP32 on the 0.5B model: a T4 FP16 trial produced a
non-finite first-step gradient and was rejected by the health gate. GRPO retains FP16 because
grouped generation is substantially more memory intensive.
"""
    ),
    code(
        """
subprocess.run(
    [sys.executable, "-m", "reasonforge.sft", "--config", "configs/sft.yaml"], check=True
)
sft_adapter = Path("outputs/sft-adapter")
assert (sft_adapter / "adapter_config.json").is_file()
sft_meta = json.loads((sft_adapter / "run_metadata.json").read_text())
sft_health = json.loads((sft_adapter / "training_health.json").read_text())
assert sft_meta["precision"] == "fp32", sft_meta
assert sft_health["nonfinite_event_count"] == 0, sft_health
print(json.dumps(sft_health, indent=2))
"""
    ),
    markdown(
        """
## 8. GPU smoke test: continue the SFT adapter with GRPO

**GPU-expensive (short):** this loads the meaningful SFT adapter as trainable PEFT weights,
runs four GRPO steps, and saves a separate smoke adapter. The initialization assertion records
the source fingerprint and trainable-parameter count.
"""
    ),
    code(
        """
subprocess.run(
    [
        sys.executable, "-m", "reasonforge.train", "--config", "configs/grpo.yaml",
        "--smoke", "--sft-adapter", "outputs/sft-adapter",
    ],
    check=True,
)
grpo_smoke = Path("outputs/sft-grpo-smoke-adapter")
assert json.loads((grpo_smoke / "initialization_assertion.json").read_text())["passed"]
"""
    ),
    markdown(
        """
## 9. Inspect GRPO group signal and enforce stop conditions

The gate checks non-finite loss/gradient events, all-zero reward variance, collapsed generations,
and high likely-truncation rate. If it stops, inspect completions and adjust temperature,
completion length, learning rate, or curriculum before attempting the 200-step run.
"""
    ),
    code(
        """
reward_diag = json.loads((grpo_smoke / "reward_diagnostics.json").read_text())
health_diag = json.loads((grpo_smoke / "training_health.json").read_text())
print(json.dumps({"reward": reward_diag, "health": health_diag}, indent=2))
assert health_diag["nonfinite_event_count"] == 0, "STOP: non-finite training metrics"
groups = reward_diag["groups"]
assert groups > 0, "STOP: no reward groups observed"
assert reward_diag["zero_total_reward_variance_groups"] / groups < 0.9, "STOP: reward signal collapsed"
assert reward_diag["mean_unique_generations"] > 1.2, "STOP: generations collapsed"
completion_count = groups * reward_diag["num_generations"]
assert reward_diag["likely_truncated_completions"] / completion_count < 0.5, "STOP: excessive truncation"
if reward_diag["groups_with_any_correct"] == 0:
    print("WARNING: smoke groups had no correct completion; structure reward varied, but inspect before continuing.")
"""
    ),
    markdown(
        """
## 10. GPU meaningful run: SFT → GRPO continuation

**GPU-expensive:** 200 optimizer steps if the smoke gate passed. The learning rate is lower than
v1, gradient clipping is explicit, truncated completions are masked, and curriculum order is
deterministic. Resume with `--resume` if necessary.
"""
    ),
    code(
        """
subprocess.run(
    [sys.executable, "-m", "reasonforge.train", "--config", "configs/grpo.yaml"], check=True
)
grpo_adapter = Path("outputs/sft-grpo-adapter")
assert (grpo_adapter / "adapter_config.json").is_file()
assert json.loads((grpo_adapter / "initialization_assertion.json").read_text())["passed"]
final_health = json.loads((grpo_adapter / "training_health.json").read_text())
final_reward_diag = json.loads((grpo_adapter / "reward_diagnostics.json").read_text())
final_meta = json.loads((grpo_adapter / "run_metadata.json").read_text())
assert final_meta["precision"] == "fp16", final_meta
assert final_health["nonfinite_event_count"] == 0, final_health
print(json.dumps({"health": final_health, "reward": final_reward_diag}, indent=2))
"""
    ),
    markdown(
        """
## 11. GPU-expensive: paired 128-example three-model evaluation

Base, SFT, and SFT+GRPO use the same deterministic official-test rows. Row artifacts preserve raw
outputs, extracted answers, exact finish metadata, all reward components, and failure categories.
Aggregate proportions include 95% Wilson intervals.
"""
    ),
    code(
        """
subprocess.run(
    [sys.executable, "-m", "reasonforge.evaluate", "--config", "configs/evaluation_v2.yaml"],
    check=True,
)
metrics_path = Path("results/sft-grpo-final/aggregate_metrics.json")
metrics = json.loads(metrics_path.read_text())
print(json.dumps(metrics, indent=2))
from IPython.display import Image, display
for plot in ("model_comparison.png", "failure_categories.png", "reward_components.png"):
    display(Image(filename=str(metrics_path.parent / plot)))
"""
    ),
    markdown(
        """
## 12. Explore in Gradio and persist artifacts

The app shows all three raw responses, parsed JSON, fallback extraction, verification, reward
components, and token-level finish metadata. The Drive cell is optional; no model weights belong
in Git.
"""
    ),
    code(
        """
# Launch and stop the cell when finished exploring:
# subprocess.run([sys.executable, "-m", "reasonforge.app", "--share"], check=True)

# Optional persistence:
# from google.colab import drive
# drive.mount('/content/drive')
# destination = Path('/content/drive/MyDrive/ReasonForge-v2-artifacts')
# destination.mkdir(parents=True, exist_ok=True)
# for source in (Path('outputs/sft-adapter'), Path('outputs/sft-grpo-adapter'), Path('results/sft-grpo-final')):
#     shutil.copytree(source, destination / source.name, dirs_exist_ok=True)
print("Training and evaluation artifacts remain in the Colab runtime until explicitly copied.")
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "accelerator": "GPU",
        "colab": {"name": "ReasonForge_Colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)
destination = Path(__file__).resolve().parents[1] / "notebooks" / "ReasonForge_Colab.ipynb"
nbf.validate(notebook)
nbf.write(notebook, destination)
print(destination)
