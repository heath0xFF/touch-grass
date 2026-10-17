---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/d1-3B/blob/da1fe36a861f24690f27f622dca1d8688503d113/LICENSE
base_model: LiquidAI/d1-3B
base_model_relation: finetune
library_name: transformers
language:
- en
tags:
- d1
- decision-model
- classification
- agents
- lora
- custom_code
- experimental
---

# touch-grass v0

**Antidoom, but for agents.**

A fine-tuned Liquid AI d1-3B decision model that scores the next action for an
agent, given its task and chronological trajectory:

`CONTINUE · RETRY · CHANGE_STRATEGY · ESCALATE · TOUCH_GRASS`

**Experimental, advisory-only model.** This is not a production safety gate,
a permission system, or an autonomous action executor. It has not been tested
on real agent deployments.

- **Code, data, tests, and reproduction:** [heath0xFF/touch-grass](https://github.com/heath0xFF/touch-grass)
- **Model format:** full merged BF16 Safetensors checkpoint, not an adapter-only release.
- **Base revision:** `LiquidAI/d1-3B@da1fe36a861f24690f27f622dca1d8688503d113`.
- **Release:** `v0`. Use the release tag or its commit hash, not a moving branch, for reproducible downloads.

Inspired by [Liquid AI's Antidoom](https://www.liquid.ai/blog/antidoom), but
**not an implementation of Antidoom or FTPO**. Antidoom targets token-level
reasoning loops. This model makes a decision over an agent trajectory.

## Actions

| Action | Meaning |
| --- | --- |
| `CONTINUE` | Useful progress; the agent incorporates evidence. |
| `RETRY` | Substantially repeating the previous action is justified by a transient or recoverable failure. |
| `CHANGE_STRATEGY` | The task remains autonomously solvable, but the current approach is unproductive. |
| `ESCALATE` | A person, credential, permission, missing fact, or unavailable capability is required. |
| `TOUCH_GRASS` | More autonomous work is likely to add cost, risk, scope, or complexity rather than accomplish the original task. |

Judge the current decision point, not the worst earlier mistake. The vocabulary
has no separate `DONE` action: work beyond a satisfied task can be `TOUCH_GRASS`.

## CLI: use the released weights

On an available, compatible CUDA GPU, with [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```sh
git clone https://github.com/heath0xFF/touch-grass.git
cd touch-grass
uv sync --locked --extra model
uv run hf download heath0xFF/touch-grass-v0 --revision v0 \
  --local-dir checkpoints/touch-grass-v0/merged
uv run touch-grass judge examples/trajectory.json --json
```

No training step is needed. Input is a JSON object with a non-empty `task`
string and a non-empty chronological list of `trajectory` strings. The CLI
rejects inputs over 2,048 tokens including the question; it does not truncate.
It returns the chosen action, all five native probabilities, and token usage.

CUDA BF16 is the default. `--device cpu` uses FP32 and requires roughly 13 GiB
for weights plus runtime memory. MPS is exposed by the CLI but was not tested.
Do not load this model beside other large GPU workloads without checking the
available memory and obtaining permission to interrupt any existing service.

## Native d1 API

**Use `system_one` or `system_one_batch`, not `generate()` or a generic text
classification pipeline.** This checkpoint preserves d1's original candidate
readout. It does not add a replacement classifier head.

The repository includes `decision.json`, containing the exact named `choice`
question used for the canonical CLI order, and `example.json`, the demo input.
This example runs with the dependencies installed above:

```python
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

path = Path(snapshot_download("heath0xFF/touch-grass-v0", revision="v0"))
device = "cuda"
model = AutoModel.from_pretrained(
    path,
    trust_remote_code=True,
    dtype=torch.bfloat16 if device == "cuda" else torch.float32,
).to(device).eval()

state = json.loads((path / "example.json").read_text())
questions = json.loads((path / "decision.json").read_text())
response = model.system_one(state, questions)
print(response["answers"]["action"]["probabilities"])
```

Loading the local snapshot also pins the tokenizer that d1 loads lazily.
Review the custom Python modules before using `trust_remote_code=True`.
For CPU, set `device = "cpu"`. The direct native API does not enforce our CLI's
2,048-token v0 limit; use the CLI for its input checks.

The original model has a larger context window and vision components. This
fine-tune and evaluation cover only short English text trajectories. Vision,
long-context behavior, and other languages were not validated.

## Measured results

Measured on DGX Spark on 2026-10-16. Both models used the same saved inputs,
candidate orders, CUDA BF16 precision, and native evaluation batch size of four.

| Set | Model | Accuracy | Macro F1 | TOUCH_GRASS recall | False stops / eligible |
| --- | --- | ---: | ---: | ---: | ---: |
| Synthetic test: 150 rows / 10 families | Base d1-3B | 64.0% | 0.638 | 9/30 | 0/90 |
| Synthetic test: 150 rows / 10 families | touch-grass v0 | 90.0% | 0.893 | 30/30 | 0/90 |
| Gold: 30 separately authored cases | Base d1-3B | 56.7% | 0.496 | 0/6 | 0/18 |
| Gold: 30 separately authored cases | touch-grass v0 | 90.0% | 0.898 | 6/6 | 1/18 (5.6%) |

A false stop is a `TOUCH_GRASS` prediction when the reference is `CONTINUE`,
`RETRY`, or `CHANGE_STRATEGY`. `ESCALATE` is excluded from that denominator.

The three tuned gold errors are important:

- `gold-03`: stopped useful staging validation instead of continuing.
- `gold-13`: retried a deterministic parser failure instead of changing strategy.
- `gold-14`: retried an expense submission with an unknown outcome instead of checking available history.

Do not use a `RETRY` decision as permission to repeat a payment or other external
write. Native probabilities are relative scores among the candidates, **not
validated estimates of real-world safety**. No post-training calibration was done.

[Full metrics, confusion matrices, and raw predictions](https://github.com/heath0xFF/touch-grass/tree/7e4dfff/results)
are published with dataset hashes. The high stop recall is only six gold cases.

## Data and training

No real agent traces were available. The synthetic dataset has 100
assistant-authored scenario templates across coding, sysadmin, DevOps,
research, and browser/tool automation. Each has 15 project-name and
candidate-order variants: **1,500 rows, not 1,500 independent cases**.

- Train: 80 families / 1,200 rows.
- Validation: 10 families / 150 rows.
- Test: 10 families / 150 rows.
- Separate gold: 30 assistant-authored cases, six per action; not human-adjudicated.

Seed 42; whole scenario families remain in one split. Synthetic splits have
balanced labels and domains. Within each family, the correct answer appears
in each candidate position exactly three times. Metadata, labels, and gold
rationales are excluded from the model state.

We followed Liquid's [official TRL decision-model recipe](https://github.com/Liquid4All/cookbook/blob/744c5b628a27adb4f219ff4a1d11e3d70259b369/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb):
use the native prompt renderer, train the correct option code plus EOS with
completion-only loss, merge language-only LoRA, then reload the native API.

| Setting | Value |
| --- | --- |
| Epochs / optimizer steps | 1 / 75 |
| Batch size / accumulation | 2 / 8, effective batch 16 |
| Learning rate / schedule | 0.0002 / cosine, 5% warmup |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Targets | Language attention, short-convolution, and MLP projections; vision frozen |
| Precision / optimizer | BF16 / PyTorch AdamW |
| Libraries | Torch 2.14.1+cu130, Transformers 5.19.0, TRL 1.15.0, PEFT 0.21.2 |
| Hardware | DGX Spark GB10, Linux ARM64, 121 GiB usable unified memory |
| Trainer runtime | 246.5 seconds |
| Peak Torch CUDA allocation | 6.51 GiB; not total host memory |

`training.json` contains the full run record. No test-based parameter search or
checkpoint selection was performed. Merged weights and native modules match
the recorded v0 artifacts; release documentation and example files are added
for distribution.

## Limits and intended use

Use this for experiments and advisory decisions. The synthetic test has only
ten independent families, and all authored sets share writing patterns.
Independent human labels, real-agent trials, prompt-injection resistance,
multilingual use, and calibrated stopping thresholds remain untested.

This release does not demonstrate reduced agent loops, token use, cost, or
incident rates. Keep hard budgets, permissions, idempotency checks, and human
approval outside the model. Never authorize destructive work from a high score.

## License and attribution

The model and derived weights use the **LFM Open License v1.0**, provided in
`LICENSE`, including its commercial-use restrictions. `NOTICE` records the
fine-tuning changes. Native d1 Python modules are unchanged from the pinned
Liquid AI checkpoint; weights were modified by language-only LoRA and merged.

The companion project's code and authored data use MIT; that does not
relicense these weights. This project is not affiliated with or endorsed by
Liquid AI.
