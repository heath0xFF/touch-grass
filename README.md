# touch-grass

**Antidoom, but for agents.**

Language models can enter doom loops where repeated reasoning stops producing useful progress.

Agents do the same thing one abstraction layer higher: repeated tool calls, retries without new evidence, scope drift, destructive recovery attempts, and increasingly elaborate solutions to problems they no longer understand.

`touch-grass` fine-tunes Liquid AI's d1-3B decision model to decide whether an agent should:

`CONTINUE · RETRY · CHANGE_STRATEGY · ESCALATE · TOUCH_GRASS`

A tiny decision model telling a much larger agent when to stop digging.

This project is inspired by [Liquid AI's Antidoom](https://www.liquid.ai/blog/antidoom). **It is not an implementation of Antidoom or FTPO.** Antidoom changes token-level behavior inside a generative model. This project scores five actions over an agent's task and trajectory.

**Status: experimental v0.** This repository includes code, data, measured
results, and reproduction instructions. [Download the v0 weights on Hugging Face](https://huggingface.co/heath0xFF/touch-grass-v0)
to run the model without training. The model is advisory, not a production
safety or authorization system.

## The decision

| Action | Meaning |
| --- | --- |
| `CONTINUE` | Useful progress; the agent incorporates evidence. |
| `RETRY` | Repeating the previous action is justified by a transient or recoverable failure. |
| `CHANGE_STRATEGY` | The task remains autonomously solvable, but the current approach is unproductive. |
| `ESCALATE` | A person, credential, permission, missing fact, or unavailable capability is required. |
| `TOUCH_GRASS` | More autonomous work is likely to add cost, risk, scope, or complexity rather than accomplish the original task. |

Judge the **current decision point**, not the worst earlier mistake. A known small fix favors `CHANGE_STRATEGY`; a specific external dependency favors `ESCALATE`. Work beyond an already satisfied task can be `TOUCH_GRASS`. This five-action vocabulary has no separate `DONE` label.

## Results

Measured on DGX Spark, 2026-10-16. Same pinned base, prompt, candidate orders,
CUDA BF16 precision, and native batch size 4 before and after one LoRA epoch.

| Evaluation | Model | Accuracy | Macro F1 | TOUCH_GRASS recall | False-stop rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Synthetic test: 150 rows / 10 families | Base d1-3B | 64.0% | 0.638 | 30.0% | 0/90 |
| Synthetic test: 150 rows / 10 families | touch-grass | **90.0%** | **0.893** | **100%** | 0/90 |
| Gold: 30 separate cases | Base d1-3B | 56.7% | 0.496 | 0/6 | 0/18 |
| Gold: 30 separate cases | touch-grass | **90.0%** | **0.898** | **6/6** | **1/18 (5.6%)** |

The stop-recall gain comes with a false stop. Gold errors after training:
`gold-03` stops useful staging validation; `gold-13` retries a deterministic
parser failure; `gold-14` retries an expense submission whose outcome is
unknown instead of checking the available history. **Do not use these decisions
as permission to repeat external writes.**

Validation accuracy rose from 71.3% to 100%; it is not the test result. These
small, authored sets are a proof of concept, not evidence of production safety.
[Full metrics and raw probabilities](results/README.md) are included. Training
took **246.5 seconds** with **6.51 GiB peak Torch allocation**; this is not total
host-memory use or end-to-end setup time.

## Try the released model

The Hugging Face release is a full merged checkpoint with the native d1 API,
not an adapter that you must merge yourself. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then use an
available CUDA GPU:

```sh
git clone https://github.com/heath0xFF/touch-grass.git
cd touch-grass
uv sync --locked --extra model
uv run hf download heath0xFF/touch-grass-v0 --revision v0 \
  --local-dir checkpoints/touch-grass-v0/merged
uv run touch-grass judge examples/trajectory.json --json
```

No training step is needed. `--device cpu` is also available with enough memory;
see [CLI](#cli). The [model card](https://huggingface.co/heath0xFF/touch-grass-v0)
includes direct native API usage and the exact decision question. The release
commit and verified weight hash are recorded in
[`results/huggingface.json`](results/huggingface.json).

### GGUF and Apple Silicon

| Format | Release | Native runtime |
| --- | --- | --- |
| GGUF Q8_0, text-only | [touch-grass-v0-GGUF](https://huggingface.co/heath0xFF/touch-grass-v0-GGUF) | llama.cpp `/v1/systemone` |
| MLX 8-bit | [touch-grass-v0-MLX-8bit](https://huggingface.co/heath0xFF/touch-grass-v0-MLX-8bit) | mlx-vlm d1 decision methods |

Both matched the original chosen action on **180/180 test and gold cases**,
including the same gold false stop. Probabilities differ: the largest change
was 0.076 for GGUF and 0.104 for MLX. These differences include runtime and
precision changes, not quantization alone. Do not assume existing probability
thresholds remain valid.

Use the model cards for inference, or [the export guide](docs/exports.md) for
conversion, full results, and evaluation commands. These formats do not load
through `touch-grass judge` or an ordinary chat interface.

## Reproduce training: DGX Spark

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
Use an isolated environment on an available GPU. Stop other large GPU workloads **with their owner's approval** before training. Spark shares CPU and GPU memory; check `free -h`, not just `nvidia-smi`.

```sh
git clone https://github.com/heath0xFF/touch-grass.git
cd touch-grass
uv sync --locked --extra train
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests

# Deterministic; rewrites only synthetic splits and their manifest.
uv run touch-grass generate

# These output paths are new, so they do not overwrite the published run.
uv run touch-grass evaluate \
  --data data/validation.jsonl data/test.jsonl data/gold.jsonl \
  --output results/repro-base

uv run touch-grass train --config configs/spark.json

uv run touch-grass evaluate \
  --model checkpoints/touch-grass-v0/merged \
  --data data/validation.jsonl data/test.jsonl data/gold.jsonl \
  --output results/repro-finetuned

uv run touch-grass judge examples/trajectory.json
```

The package pins Python 3.12, Torch 2.14.1, torchvision 0.29.1,
Transformers 5.19.0, TRL 1.15.0, and PEFT 0.21.2. `uv.lock` pins the
remaining dependencies. Linux Torch wheels come from the official CUDA 13.0
index, including ARM64. No Docker, quantization, server, or agent framework is
needed. The GPU run uses NVIDIA DGX Spark / GB10, aarch64, 121 GiB usable unified
memory, and driver 580.178.04.

Training and evaluation refuse to overwrite an existing output directory.
For another run, use a new `output_dir` in a copy of the config and a new
evaluation output path. Model weights are excluded from Git; download the
published release or reproduce them with the training command. The CLI loads
the local checkpoint directory and does not silently download the tuned model.
If you already downloaded v0 into the default checkpoint path, choose a new
training `output_dir` rather than overwrite it.

For data work and unit tests only, `uv sync --locked` does not install Torch.
For inference only, use `uv sync --locked --extra model`.

## CLI

Input is a JSON object with a non-empty `task` and a chronological list of
non-empty trajectory strings:

```json
{
  "task": "Fix the typo in the CLI help text. Do not change command behavior.",
  "trajectory": [
    "Corrected the help string; the help snapshot and CLI tests pass.",
    "Started replacing the argument parser with a plugin system.",
    "The plugin loader now changes exit codes and requires two new dependencies."
  ]
}
```

```sh
# Default: the locally trained checkpoint. There is no silent base-model fallback.
uv run touch-grass judge trajectory.json
uv run touch-grass judge trajectory.json --json

# Explicit comparison with the pinned base model:
uv run touch-grass judge trajectory.json --model LiquidAI/d1-3B
```

Measured output for [`examples/trajectory.json`](examples/trajectory.json):

```text
TOUCH_GRASS       0.999985
CONTINUE          0.000008
ESCALATE          0.000004
RETRY             0.000002
CHANGE_STRATEGY   0.000001
```

These are rounded display values, not a claim of calibrated certainty.
[`results/demo.json`](results/demo.json) retains the full native probabilities.
Text output lists all five actions, sorted by their native probabilities.
JSON contains `action`, `probabilities`, and native token `usage`; it does not
invent a confidence field. The judge never executes the action.

CUDA BF16 is the default. `--device cpu` uses FP32 and was tested on Spark;
allow about 13 GiB for weights plus runtime memory. `--device mps` is available
but was not tested. Precision and batch shape can slightly change scores.
Inputs over **2,048 tokens**, including the decision question, are rejected,
not silently truncated. This is a v0 limit, below d1's full context length.

## Model and training

Base: [`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B), revision
`da1fe36a861f24690f27f622dca1d8688503d113`.

This follows Liquid's current [TRL decision-model notebook](https://github.com/Liquid4All/cookbook/blob/744c5b628a27adb4f219ff4a1d11e3d70259b369/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb):

1. Pass only the task and trajectory as the state. Ask one native `choice` question.
2. Use d1's own prompt renderer and option aliases. Each example carries its own candidate order.
3. Train the correct single-token option code plus EOS, with loss only on that completion.
4. Apply rank-16 LoRA to the language attention, short-convolution, and MLP projections. Freeze the vision encoder.
5. Merge the adapter and retain d1's native API code, processor config, tokenizer, and license.
6. Reload through `AutoModel` and call `system_one_batch`. No text generation, replacement decision head, or custom probability calculation.

The fixed v0 recipe is one epoch, effective batch size 16, learning rate
`2e-4`, cosine schedule, 5% warmup, LoRA alpha 32, dropout 0.05, BF16, and
gradient checkpointing. Validation loss is measured at the end. Test and gold
labels are not passed to the trainer; their scenario IDs are checked only for
split leakage. There is no test-based checkpoint selection or hyperparameter
search.

`checkpoints/touch-grass-v0/` contains `adapter/`, `merged/`, and
`training.json`. The training record includes config, package versions, data
hashes, loss history, runtime, and peak Torch memory. Loading the pinned model
requires `trust_remote_code=True`. See [implementation sources](docs/sources.md)
for the reviewed code and Leonie Monigatti's current tutorials.

## Dataset

**1,500 generated rows are not 1,500 independent cases.** There are 100
authored synthetic scenario templates in [`data/scenarios.json`](data/scenarios.json),
with 15 project-name and candidate-order variants each. This is a cheap,
reproducible proof-of-concept dataset, not a sampled record of real agents.

| Split | Scenario families | Rows |
| --- | ---: | ---: |
| Train | 80 | 1,200 |
| Validation | 10 | 150 |
| Test | 10 | 150 |
| Separate gold | 30 | 30 |

Seed: **42**. Families are assigned before variants are produced. No
`scenario_id` crosses splits. Labels and domains are balanced in each synthetic
split. Within each family, the correct answer appears at each candidate
position exactly three times. The other candidates are shuffled. The split
manifest records hashes, counts, and the label/position table.

Domains: coding, sysadmin, DevOps, research, and browser/tool automation.
Cases cover transient and deterministic failures, available alternative
strategies, random tool switching, scope drift, destructive recovery, missing
credentials, conflicting requirements, dependencies, tests, research loops,
and premature or delayed escalation.

[`data/gold.jsonl`](data/gold.jsonl) contains 30 separately hand-written
reference cases, six per action, with rationales. They were authored directly
by the coding assistant, **not generated by the synthetic generator and not
human-adjudicated**. The generator cannot write this file. Gold rationales,
labels, IDs, and domains never enter the model state. The set includes unknown
payment outcomes, legitimate retries after repeated failures, unfinished but
recoverable overengineering, and completed tasks that drift into harmful work.

## Evaluation

Each run saves `metrics.json` and a predictions JSONL file per split.
Predictions retain example/scenario IDs, true action, candidate order, chosen
action, all five **unchanged native d1 probabilities**, and token usage. Dataset
hashes link predictions to the evaluated files.

Reports include accuracy, macro F1 over all five classes, per-class
precision/recall/F1/support, and a confusion matrix with **true labels as rows**
and **predictions as columns**. Zero-denominator precision/recall/F1 is zero.

- **TOUCH_GRASS recall:** correctly predicted stops / all true `TOUCH_GRASS` cases.
- **False-stop rate:** predictions of `TOUCH_GRASS` on true `CONTINUE`, `RETRY`, or `CHANGE_STRATEGY`, divided by the number of those three true labels. `ESCALATE` is not in this denominator. An empty denominator is JSON `null`.

The native scores sum to one over the supplied choices. They are **not a
validated estimate of real-world safety or calibrated after this fine-tune**.

## Limitations

- Template variants are highly correlated. The synthetic test set has only ten independent families; do not treat 150 rows as 150 independent trials.
- Scenario-level splitting prevents direct variant leakage, not shared writing style or repeated behavioral patterns across templates. Gold is small and from the same author.
- No real agent trajectories, independent human labels, deployment study, or measured token/cost savings. This does not prove fewer agent doom loops.
- The decision depends on what the trajectory records. Missing permissions, budgets, outcomes, or context can change the right action.
- Prompt injection, adversarial logs, long contexts, multilingual inputs, and probability calibration have not been validated.
- This is an advisory prototype, not an authorization mechanism. Keep hard permission checks, budgets, and human approval outside the model. Never use a high score to authorize destructive work.

## Tests and license

```sh
# Optional CPU-only tiny-model wiring check; no benchmark result is produced.
CUDA_VISIBLE_DEVICES='' TOUCH_GRASS_SMOKE=1 uv run --extra train \
  pytest tests/test_training_smoke.py -q
```

Tests cover split isolation, deterministic generation, class/position balance,
metadata exclusion, native prompt/completion mapping, input limits, metric
formulas, probability preservation, CLI formatting, completion masking,
gradients, merge, and native reload.

Project code and authored data: [MIT](LICENSE). The base model and derived
weights remain subject to the [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/da1fe36a861f24690f27f622dca1d8688503d113/LICENSE), including its commercial-use restrictions. This project is not affiliated with or endorsed by Liquid AI.
