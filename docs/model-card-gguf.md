---
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/d1-3B/blob/da1fe36a861f24690f27f622dca1d8688503d113/LICENSE
base_model: heath0xFF/touch-grass-v0
base_model_relation: quantized
language:
- en
tags:
- gguf
- llama-cpp
- d1
- decision-model
- experimental
---

# touch-grass v0 — GGUF Q8_0

**Antidoom, but for agents.** A text-only Q8_0 export of
[the merged touch-grass v0 checkpoint](https://huggingface.co/heath0xFF/touch-grass-v0),
fine-tuned from Liquid AI d1-3B. It scores five next actions:

`CONTINUE · RETRY · CHANGE_STRATEGY · ESCALATE · TOUCH_GRASS`

**Experimental and advisory only.** Not a safety gate or authorization system.
This is a decision model, not a chat model that generates action labels.
Use llama.cpp's native **`/v1/systemone`** API. Ordinary chat clients are not
necessarily compatible.

## Run

Use a d1-capable llama.cpp build. Tested with `0.7.0-dev`, commit
[`10b8b7553494f3638da7668cf17cac089229122a`](https://github.com/ggml-org/llama.cpp/commit/10b8b7553494f3638da7668cf17cac089229122a),
on Apple M5 Pro with Metal. Older builds without `lfm2-d1` support will fail.

With the Hugging Face CLI installed:

```sh
hf download heath0xFF/touch-grass-v0-GGUF --revision v0 --local-dir touch-grass-v0-GGUF
llama-server -m touch-grass-v0-GGUF/touch-grass-v0-Q8_0.gguf \
  --host 127.0.0.1 --port 18081 --ctx-size 2048 --parallel 1 \
  --batch-size 2048 --ubatch-size 2048 --no-context-shift --no-webui -ngl 99
```

In another terminal:

```sh
curl --fail-with-body http://127.0.0.1:18081/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @touch-grass-v0-GGUF/request.json
```

`request.json` includes a demo task, chronological trajectory, and the exact
five-action question. Replace its `state` to judge another trajectory; retain
the question for comparable behavior. The response includes `choice`, all five
native candidate probabilities, and zero output tokens. Do not interpret native
confidence fields or probabilities as calibrated real-world safety estimates.

Only text was exported: **no vision tower or multimodal projector is included**.
Keep the complete rendered input within v0's 2,048-token limit. The companion
project's PyTorch CLI does not load GGUF; use the native endpoint above.
Do not expose this unauthenticated example server beyond loopback.

## Measured checks

Compared with the published CUDA BF16 checkpoint, using the same input files
and per-case candidate orders. Both are held-out sets, not quantization data.

| Set | Accuracy | Macro F1 | Stop recall | False stops | Same top action as BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic test: 150 rows / 10 families | 90.0% | 0.893 | 30/30 | 0/90 | 150/150 |
| Gold: 30 separately authored cases | 90.0% | 0.898 | 6/6 | 1/18 | 30/30 |

The largest candidate-probability change is **0.0413 on test** and **0.0757
on gold**. Mean absolute changes are 0.000266 and 0.001392 respectively.
Differences include runtime, precision, batch shape, and quantization effects;
this is not a measurement of quantization alone. Matching top actions does not
mean matching probabilities or calibration. The original three gold errors remain.

False stops count `TOUCH_GRASS` predictions on true `CONTINUE`, `RETRY`, or
`CHANGE_STRATEGY` cases. The gold set is assistant-authored, not human-adjudicated.
Synthetic variants are correlated. No real-agent safety validation was done.

[Raw predictions and metrics](https://github.com/heath0xFF/touch-grass/tree/main/results/gguf-q8_0)
· [Conversion and evaluation recipe](https://github.com/heath0xFF/touch-grass/blob/main/docs/exports.md)
· [Original training and limitations](https://huggingface.co/heath0xFF/touch-grass-v0)

## Provenance and license

Source release: `heath0xFF/touch-grass-v0@8cd6d51892d82b61db40f2ff4889a7dfe0755f89`.
Converted through F16 GGUF, then quantized with `llama-quantize Q8_0`. No
retraining, calibration dataset, replacement head, or generated-label scoring.
The GGUF includes the native d1 decision type and System One prompt template.
`export.json` records the tool revisions, source hash, and artifact hashes.

Weights use the **LFM Open License v1.0**, including its commercial-use
restrictions. `LICENSE` and `NOTICE` are included. The companion code's MIT
license does not relicense the weights. Not affiliated with or endorsed by
Liquid AI.
