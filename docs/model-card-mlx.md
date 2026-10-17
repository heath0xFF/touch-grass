---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/d1-3B/blob/da1fe36a861f24690f27f622dca1d8688503d113/LICENSE
base_model: heath0xFF/touch-grass-v0
base_model_relation: quantized
library_name: mlx
language:
- en
tags:
- mlx-vlm
- d1
- decision-model
- experimental
---

# touch-grass v0 — MLX 8-bit

**Antidoom, but for agents.** An Apple Silicon export of
[the merged touch-grass v0 checkpoint](https://huggingface.co/heath0xFF/touch-grass-v0),
fine-tuned from Liquid AI d1-3B. It scores five next actions:

`CONTINUE · RETRY · CHANGE_STRATEGY · ESCALATE · TOUCH_GRASS`

**Experimental and advisory only.** Not a safety gate or authorization system.
Use mlx-vlm's native d1 decision methods, not text generation or a generic chat
interface. No replacement classifier head is added.

## Run on Apple Silicon

Tested on Apple M5 Pro with MLX 0.32.4 and mlx-vlm 0.7.8, commit
[`73d366ecf9233b3fc6549531a7f2386726d09ff6`](https://github.com/Blaizzy/mlx-vlm/commit/73d366ecf9233b3fc6549531a7f2386726d09ff6).
mlx-vlm versions without d1 support will not work.

Save the following as `judge_mlx.py`:

```python
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_vlm import load

path = Path(snapshot_download("heath0xFF/touch-grass-v0-MLX-8bit", revision="v0"))
model, processor = load(path)
request = json.loads((path / "request.json").read_text())
question = request["questions"]["action"]
rows, tokens = model.probabilities(processor, request["state"], [question])
probabilities = dict(zip(question["criteria"], rows[0], strict=True))
print({"action": max(probabilities, key=probabilities.get), "probabilities": probabilities})
```

Run with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
uv run --with 'mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git@73d366ecf9233b3fc6549531a7f2386726d09ff6' \
  --with 'mlx==0.32.4' --with 'transformers==5.19.0' python judge_mlx.py
```

`request.json` includes a demo task, chronological trajectory, and the exact
five-action question. Replace its `state` to judge another trajectory; retain
the question for comparable behavior. The code uses the native, unrounded
`model.probabilities` readout. The higher-level `mlx_vlm.predict` API also works,
but rounds its probabilities to four decimal places and calls the selected
label `value`. Neither API generates output tokens for these decisions.

Keep the complete rendered input within v0's **2,048-token limit**; the native
MLX method does not enforce this project-specific limit. The evaluation script
checks it. The companion project's PyTorch CLI does not load these MLX weights.

## Measured checks

Compared with the published CUDA BF16 checkpoint, using the same input files
and per-case candidate orders. Both are held-out sets, not quantization data.

| Set | Accuracy | Macro F1 | Stop recall | False stops | Same top action as BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic test: 150 rows / 10 families | 90.0% | 0.893 | 30/30 | 0/90 | 150/150 |
| Gold: 30 separately authored cases | 90.0% | 0.898 | 6/6 | 1/18 | 30/30 |

The largest candidate-probability change is **0.0371 on test** and **0.1045
on gold**. Mean absolute changes are 0.000351 and 0.002098 respectively.
Differences include runtime, precision, batch shape, and quantization effects;
this is not a measurement of quantization alone. Matching top actions does not
mean matching probabilities or calibration. The original three gold errors remain.

False stops count `TOUCH_GRASS` predictions on true `CONTINUE`, `RETRY`, or
`CHANGE_STRATEGY` cases. The gold set is assistant-authored, not human-adjudicated.
Synthetic variants are correlated. No real-agent safety validation was done.
Never treat a high probability as permission to repeat a payment or take a
destructive action. No post-training probability calibration was performed.

[Raw predictions and metrics](https://github.com/heath0xFF/touch-grass/tree/main/results/mlx-8bit)
· [Conversion and evaluation recipe](https://github.com/heath0xFF/touch-grass/blob/main/docs/exports.md)
· [Original training and limitations](https://huggingface.co/heath0xFF/touch-grass-v0)

## Provenance and license

Source release: `heath0xFF/touch-grass-v0@8cd6d51892d82b61db40f2ff4889a7dfe0755f89`.
Converted with mlx-vlm: 8-bit affine quantization, group size 64, with other
tensors in float16. Language linear/embedding weights are quantized; the vision
tower and projector remain float16. Vision weights are retained but **vision
use was not tested**. No retraining or calibration dataset was used.

The upstream PyTorch modules are not bundled. `config.json` keeps its `D1Model`
marker so mlx-vlm selects its built-in native d1 implementation. These are MLX
weights, not a checkpoint to load through Transformers. `export.json` records
the tool revisions, source hash, and artifact hashes.

Weights use the **LFM Open License v1.0**, including its commercial-use
restrictions. `LICENSE` and `NOTICE` are included. The companion code's MIT
license does not relicense the weights. Not affiliated with or endorsed by
Liquid AI.
