# Implementation sources

Checked on 2026-10-16, before model-specific implementation.

- [Liquid's d1-3B model card](https://huggingface.co/LiquidAI/d1-3B): a state and typed questions go into one forward pass; `system_one` returns candidate probabilities. The generic Hub `generate()` examples are not the decision API.
- [Open d1 release](https://huggingface.co/blog/LiquidAI/open-d1): model architecture, native API, and Transformers requirements.
- [Official TRL decision-model notebook](https://github.com/Liquid4All/cookbook/blob/744c5b628a27adb4f219ff4a1d11e3d70259b369/finetuning/notebooks/sft_for_decision_models_with_trl.ipynb): reuse `prompt.render`, `as_question`, and `aliases`; train option code + EOS with completion-only loss; apply language-only LoRA; merge and copy the native API code and license.
- [Official Unsloth decision-model notebook](https://github.com/Liquid4All/cookbook/blob/744c5b628a27adb4f219ff4a1d11e3d70259b369/finetuning/notebooks/sft_for_decision_models_with_unsloth.ipynb): explicitly warns that `FastDecisionModel` would add a new head and lose d1's trained readout and System One API. This project does not use that path.
- [Liquid's TRL documentation](https://docs.liquid.ai/lfm/fine-tuning/trl): general fine-tuning guidance. The d1-specific notebook takes precedence here.
- [Leonie Monigatti's 2026-10-16 post](https://x.com/helloiamleonie/status/2110994738811715943): introduces the current TRL and Unsloth d1-3B tutorials. Both official notebooks were inspected. Her LinkedIn post was indexed, but direct page access was unavailable.
- [Antidoom explanation](https://www.liquid.ai/blog/antidoom) and [implementation](https://github.com/Liquid4All/antidoom): identify the token that begins a reasoning loop and apply Final Token Preference Optimization. That is the inspiration, not this project's training objective.

## Pinned model code

Base model: `LiquidAI/d1-3B`, revision
`da1fe36a861f24690f27f622dca1d8688503d113`.

The implementation was checked directly: `modeling_d1.py`, `api.py`,
`prompt.py`, `runner.py`, and the backbone modules. The native choice readout
uses the best supported token form for each candidate, then normalizes over
candidates. We call that implementation unchanged. We do not replace it with
sequence likelihoods, a new head, generated labels, or a second softmax.

Both training and inference load a local snapshot of the pinned revision. This
also pins the tokenizer loaded lazily by d1's engine. Merged checkpoints retain
the original native Python modules and model license. Loading these models
requires `trust_remote_code=True`; review the pinned code before use.

## Training choice

TRL 1.15.0 + PEFT 0.21.2 + Transformers 5.19.0 matches the current official
TRL notebook. Torch 2.14.1 CUDA 13.0 has Linux ARM64 wheels for the Spark;
no Unsloth patches, quantization, new decision head, or container is needed.
The optimizer is standard PyTorch AdamW. Vision weights remain frozen.

The optional small CPU test checks formatting, completion masking, gradients,
merge, and native reload. TRL 1.15's fused loss kernel requires CUDA, so that
CPU check uses the backbone's unfused loss for its one update. It is a wiring
test with randomly initialized tiny weights, **not** a d1-3B fine-tuning result.
