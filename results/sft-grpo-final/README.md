# ReasonForge v2 measured artifacts

These artifacts come from one seed-42 Tesla T4 run of the checked-in SFT → GRPO pipeline. The
evaluation uses the same 128 examples from GSM8K's official test split for the base, SFT, and
SFT+GRPO policies with greedy decoding and a 256-token completion cap. Training used only the
official training split; the saved SFT manifest records zero official-test examples used and zero
train/validation overlap.

| Model | Math correct | Math accuracy (Wilson 95% CI) | Strict E2E | JSON valid | Truncated |
|---|---:|---:|---:|---:|---:|
| Base | 12/128 | 9.4% (5.4–15.7%) | 0/128 | 11.7% | 44.5% |
| SFT | 21/128 | 16.4% (11.0–23.8%) | 21/128 | 96.1% | 3.1% |
| SFT + GRPO | 25/128 | 19.5% (13.6–27.2%) | 23/128 | 99.2% | 0.8% |

The SFT and SFT+GRPO intervals overlap. The four-answer difference is the measured outcome of this
run, not evidence of a statistically established general improvement.

## Artifact map

- [`per_example.jsonl`](per_example.jsonl) and [`per_example.csv`](per_example.csv): all 384 raw
  paired rows, independent math extraction, structured verification, reward components, exact
  finish reason, completion-token count, truncation flag, and failure categories. JSONL is the
  authoritative raw-text artifact; CSV represents embedded newlines as the two characters `\n`.
- [`aggregate_metrics.json`](aggregate_metrics.json): proportions, two-sided 95% Wilson intervals,
  reward means, response lengths, and primary failure counts.
- [`evaluation_manifest.json`](evaluation_manifest.json): model, adapter, dataset, split, and
  decoding provenance.
- [`validation_report.json`](validation_report.json): independent row-count, pairing, aggregate,
  interval, finish-metadata, finite-number, leakage-boundary, and lineage checks.
- [`representative_examples.json`](representative_examples.json): bounded examples selected from
  the complete row artifact.
- [`model_comparison.png`](model_comparison.png),
  [`failure_categories.png`](failure_categories.png), and
  [`reward_components.png`](reward_components.png): generated plots.
- [`training/`](training/): SFT/GRPO health records, trainer states, resolved configurations,
  adapter-lineage assertion, dataset fingerprint, raw group reward diagnostics, derived signal
  and stability summaries, and hardware record.

No dataset cache, base-model weights, LoRA weights, optimizer checkpoint, or secret is included.
