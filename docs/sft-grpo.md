# SFT → GRPO training notes

ReasonForge v2 uses a two-stage LoRA pipeline:

1. `reasonforge.sft` converts verified GSM8K `<<expression=result>>` annotations into the exact
   response schema and trains `outputs/sft-adapter` with completion-only loss.
2. `reasonforge.train` validates and fingerprints that adapter, reloads it with
   `PeftModel.from_pretrained(..., is_trainable=True)`, and continues the same weights with GRPO
   into `outputs/sft-grpo-adapter`.

The GRPO stage refuses an adapter without SFT metadata, a matching base-model ID, configuration,
and weights. It also refuses to use the source directory as its output directory. The generated
`initialization_assertion.json` is the machine-readable lineage record.

## Objective curriculum

The fixed stages are simple arithmetic, one-step word problems, fraction/percentage/equation
problems, then multi-step reasoning. Classification depends only on observable question text and
verified GSM8K calculation annotations. SFT sorts by stage and uses a sequential sampler. GRPO
sorts training records the same way and disables trainer shuffling. Set `curriculum.max_stage` to
1–4 to inspect or cap the active curriculum.

## Stability gate

Both stages set explicit gradient clipping and preserve rather than filter non-finite logs. SFT
uses FP32 on the 0.5B model after a T4 FP16 trial exposed a first-step non-finite gradient;
GRPO retains FP16 for its more memory-intensive grouped generation.
`training_health.json` reports finite metric ranges and every observed NaN/Inf. AMP overflow skips
are recorded as unknown because TRL/Trainer callbacks do not expose them reliably in this pinned
stack. GRPO additionally writes `reward_diagnostics.json`, including groups with a correct sample,
all-incorrect groups, zero correctness/reward variance, mean structure reward, unique generations,
and likely text truncation. The Colab notebook stops on non-finite metrics, collapsed total reward,
collapsed generations, or excessive truncation.

## Commands

```bash
python -m reasonforge.sft --config configs/sft.yaml --smoke
python -m reasonforge.sft --config configs/sft.yaml
python -m reasonforge.train --config configs/grpo.yaml --smoke
python -m reasonforge.train --config configs/grpo.yaml
python -m reasonforge.evaluate --config configs/evaluation_v2.yaml
```

All model weights, adapters, checkpoints, caches, and downloaded datasets are ignored by Git.
Only compact measured evaluation artifacts explicitly allowlisted under `results/` are published.

## Measured T4 run

The accepted seed-42 run used a Tesla T4. SFT completed 100 FP32 steps in 343.6 seconds with train
loss 0.2752, final evaluation loss 0.2162, and no non-finite events in 105 logged records. GRPO
continued the fingerprinted SFT adapter for 200 FP16 steps in 1,823.1 seconds with train loss
0.0564 and no non-finite events in 201 records. Of 200 eight-generation groups, 129 contained a
correct answer, 71 were all incorrect, ten had zero total-reward variance, mean reward variance was
4.506, and mean unique completions was 6.48.

The paired 128-example official-test evaluation measured math accuracy of 9.4% for base, 16.4% for
SFT, and 19.5% for SFT+GRPO. Exact counts, Wilson intervals, raw outputs, finish metadata, plots,
and failure categories are in [`results/sft-grpo-final/`](../results/sft-grpo-final/). The SFT and
SFT+GRPO intervals overlap; treat their difference as a measured single-run result, not proof of a
general GRPO effect.
