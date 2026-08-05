# ReasonForge

ReasonForge is a reinforcement-learning sandbox for the claim: can GRPO post-training make a 0.5B instruction model produce concise, machine-verifiable arithmetic solutions more reliably? It trains `Qwen/Qwen2.5-0.5B-Instruct` with LoRA on GSM8K prompts and rewards structured JSON, exact arithmetic, reference-answer correctness, and consistency between the final calculation and answer.

## How it works

Group Relative Policy Optimization (GRPO) samples several completions for the same prompt, scores them, and turns their within-group relative quality into an advantage signal. A learned value model is not required. Better completions in a group are reinforced, while a KL term can constrain drift from the reference policy. With four generations, a prompt can produce both malformed and correct candidates; the verifier supplies a dense ranking among them even when the absolute task is hard.

```mermaid
flowchart LR
    D["GSM8K train split"] --> S["Deterministic train/validation split"]
    T["GSM8K held-out test"] --> E["Paired evaluation"]
    S --> P["System + user chat prompt"]
    P --> M["Qwen2.5 0.5B + LoRA"]
    M --> G["4 sampled completions"]
    G --> J["JSON + Pydantic validation"]
    G --> V["Restricted AST + SymPy verifier"]
    J --> R["Component rewards"]
    V --> R
    R --> O["GRPO policy update"]
    O --> M
    B["Untouched base model"] --> E
    A["Base + saved adapter"] --> E
    E --> X["CSV, JSONL, metrics, plots, examples"]
```

The arithmetic parser accepts only numeric constants, parentheses, unary signs, and `+ - * / **` (with `^` normalized to exponentiation). It limits input length, AST size, exponent size, literal size, and result magnitude. Names, calls, attributes, assignments, imports, containers, and unsupported characters are rejected before expressions are rebuilt as SymPy objects. Python `eval` is never used.

## Response contract

```json
{
  "method": "unit-rate multiplication",
  "calculations": [
    {
      "expression": "12 * 5",
      "result": "60"
    }
  ],
  "final_answer": "60"
}
```

`method` is an unverified short label. Each item in `calculations` is checked in the restricted arithmetic language. The last valid calculation result must equal `final_answer` for the consistency reward.

## Reward design

| Component | Maximum | Encourages |
|---|---:|---|
| Schema | 1.0 | Exact JSON and strict Pydantic schema; merely parseable JSON earns 0.25 |
| Answer correctness | 5.0 | Mathematical equivalence to the held-out reference |
| Calculation validity | 2.0 | Fraction of structured calculations that evaluate to their claimed results |
| Final consistency | 1.5 | Last verified result equals `final_answer` |
| Conciseness | 0.5 | A bounded response, awarded only when the answer is correct |
| Suspicious output | −1.0 | Explicit penalty for obvious code-execution/injection-like tokens |

This weighting makes mathematical correctness dominate presentation. The tests enforce:

```text
correct answer + correct structure
  > correct answer + malformed structure
  > wrong answer + correct structure
  > wrong answer + malformed structure
```

TRL receives the components as separately named reward functions and logs each component plus the trainer's summed reward.

## Setup

Python 3.11 or 3.12 is recommended. A clean local setup is:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For NVIDIA training, install the optional 8-bit optimizer support:

```bash
python -m pip install -e '.[dev,gpu]'
```

Dependencies are pinned around the documented TRL 0.24.0 interface: TRL requires Transformers ≥4.56.1, Datasets ≥3.0.0, and Accelerate ≥1.4.0; its PEFT extra requires PEFT ≥0.8.0. The exact resolved choices are in `pyproject.toml` and `requirements-colab.txt`.

## Dataset and training

The official GSM8K test split is reserved for final comparison. Validation comes only from the official training split. Shuffling, splitting, and subset selection are seeded. Set a subset size to `null` in YAML to use all available examples.

```bash
python -m reasonforge.dataset --config configs/training.yaml
python -m reasonforge.train --config configs/training.yaml
```

The default is an exploratory T4 configuration: 512 train prompts, 100 GRPO steps, four generations, effective single-process batch size four, FP16 on CUDA, gradient checkpointing, and LoRA rank 16. If memory is tight:

```bash
python -m reasonforge.train --config configs/training.yaml --fallback
```

The fallback uses two generations and a shorter completion limit. When `bitsandbytes` and CUDA are detected, training uses paged AdamW 8-bit; otherwise it falls back to torch AdamW. Checkpoints are retained and resumable:

```bash
python -m reasonforge.train --config configs/training.yaml --resume
# or: --resume outputs/reasonforge-adapter/checkpoint-50
```

The Qwen LoRA adapter targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`: attention and MLP linear projections with most of the model's adaptable computation, while embeddings and the language-model head remain frozen. The final directory includes the adapter, tokenizer, trainer files, and `run_metadata.json` with the resolved configuration, library versions, seed, and hardware.

### Google Colab

Open [`notebooks/ReasonForge_Colab.ipynb`](notebooks/ReasonForge_Colab.ipynb) in a T4 GPU runtime. It clones this repository by default; change `REPO_URL` only when running a fork, or upload/open the project at `/content/ReasonForge`. Run the installation, data, and smoke-test cells first. The installation cell removes Colab's unused `torchvision`/`torchaudio` wheels when they conflict with the pinned text-only training stack. The notebook labels costly training and evaluation cells, provides a three-step demonstration configuration, keeps the 100-step exploratory configuration, launches Gradio, and optionally copies artifacts to Google Drive.

A T4 is the target, not a guarantee: available VRAM and Colab's preinstalled CUDA stack vary. Use `--fallback`, fewer examples, or a shorter completion length if the runtime runs out of memory.

## Evaluation

Evaluation runs the untouched base model and the same base model with the trained adapter on identical official-test examples, prompts, deterministic decoding settings, and seed:

```bash
python -m reasonforge.evaluate --config configs/evaluation.yaml
```

It writes:

- `results/latest/per_example.csv` and `per_example.jsonl`
- `aggregate_metrics.json`
- `comparison.png`
- `representative_examples.json`

Metrics include answer accuracy, valid JSON, schema compliance, calculation validity, final consistency, average total reward, response length, and failure counts. To insert real metrics into the generated block below:

```bash
python -m reasonforge.evaluate --config configs/evaluation.yaml --update-readme
# Or reuse an existing artifact without rerunning models:
python scripts/update_readme_results.py --metrics results/latest/aggregate_metrics.json
```

### Generated results

The first full run used the checked-in training configuration: seed 42, 512 GSM8K training prompts, 128 validation prompts, four sampled completions per group, 100 optimizer steps, LoRA rank 16, FP16, gradient checkpointing, and paged AdamW 8-bit. It completed on a Tesla T4 (15,360 MiB) in 1,395.7 seconds. Mean training loss was 0.09177; all 100 logged losses were finite. Gradient norm was finite for 99/100 steps: step 10 logged `NaN`, after which the run recovered and completed 90 further finite-gradient steps. This isolated anomaly is preserved in the trainer state and prevents an all-finite training claim.

The paired evaluation used 32 examples selected deterministically from the untouched official GSM8K test split, seed 42, greedy decoding, and a 192-token completion cap. The same rows and settings were used for the untouched base model and the base model plus the saved adapter.

<!-- RESULTS:START -->
| Model | Accuracy | Valid JSON | Schema | Calc validity | Consistency | Avg reward |
|---|---:|---:|---:|---:|---:|---:|
| aligned | 0.0% | 75.0% | 40.6% | 6.2% | 6.2% | 0.727 |
| base | 0.0% | 6.2% | 3.1% | 0.0% | 0.0% | 0.047 |
<!-- RESULTS:END -->

The adapter substantially improved JSON formatting, schema compliance, average verifier reward, and concision (mean response length fell from 612.3 to 336.7 characters), but it did **not** improve answer accuracy: both conditions were 0/32. More aligned outputs were parseable, so the verifier could classify more of their downstream calculation, consistency, and wrong-answer failures; those larger classified-failure counts should not be interpreted independently as regressions.

![Base-versus-aligned measured comparison](results/first-gpu-run/comparison.png)

The complete measured record is in [`results/first-gpu-run/`](results/first-gpu-run/): paired row-level outputs, aggregate metrics, representative failures, plot, exact training and evaluation YAML, hardware and library metadata, artifact manifest, and the 100-step trainer state. Adapter/model weights are intentionally excluded. One representative aligned response was valid JSON with a fully verified arithmetic trace and a consistent final answer, yet answered the word problem incorrectly; this is a concrete example of why arithmetic self-consistency is not equivalent to correct problem interpretation.

## Interactive comparison

```bash
python -m reasonforge.app --adapter-path outputs/reasonforge-adapter
```

Open <http://127.0.0.1:7860>. Enter a problem and, ideally, its reference answer. The app lazily loads models, shows base and aligned responses side by side, exposes parsed JSON, calculation-level verification, and each reward component. If no trained adapter exists, it gives training instructions instead of failing at import or startup.

## Development checks

CPU-friendly tests do not download GSM8K or model weights:

```bash
pytest
ruff check .
ruff format --check .
python -m reasonforge.dataset --help
python -m reasonforge.train --help
python -m reasonforge.evaluate --help
python -m reasonforge.app --help
python -c "import nbformat; nbformat.validate(nbformat.read('notebooks/ReasonForge_Colab.ipynb', 4))"
```

The GitHub Actions workflow installs only the verifier's CPU test dependencies, not PyTorch, Transformers, model weights, or GSM8K. Release validation completed on August 4, 2026: 42 tests passed; Ruff lint and formatting passed; the Colab notebook validated; all four CLI entry points loaded; offline imports passed; the source distribution and wheel built; `git diff --check` passed; and the CPU validation workflow completed successfully.

## Project structure

```text
configs/                 Training and evaluation YAML
notebooks/               End-to-end Colab workflow
src/reasonforge/         Dataset, verifier, rewards, training, evaluation, and app
tests/                   CPU-only unit and adversarial tests
results/first-gpu-run/   Tracked measurements and metadata from the first T4 run
results/                 Other generated evaluation artifacts (ignored by Git)
pyproject.toml            Package metadata, pins, lint/test configuration
requirements-colab.txt   Colab dependency lock
```

## License

ReasonForge is released under the [MIT License](LICENSE). Dataset and model artifacts retain their respective upstream licenses and terms.
