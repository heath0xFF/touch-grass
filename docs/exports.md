# GGUF and MLX exports

Two 8-bit exports of the published v0 checkpoint. No retraining, calibration
data, custom classifier head, or generated-label scoring.

| Release | Weight file | Runtime |
| --- | ---: | --- |
| [GGUF Q8_0](https://huggingface.co/heath0xFF/touch-grass-v0-GGUF) | about 2.87 GB | llama.cpp native `/v1/systemone` |
| [MLX 8-bit](https://huggingface.co/heath0xFF/touch-grass-v0-MLX-8bit) | about 3.72 GB | mlx-vlm native d1 decision methods |

Use each model card for download and inference instructions. These are not
ordinary chat-model releases. The existing `touch-grass judge` command still
loads the original PyTorch checkpoint; it does not auto-detect these formats.

GGUF is **text-only**, without a vision projector. MLX keeps the vision tower
and projector in float16, but vision was not tested. Both retain the LFM Open
License v1.0 and modification notices. The project's MIT license does not
relicense derived weights.

## Results and probability changes

Measured on Apple M5 Pro with 48 GiB unified memory, using Metal and one case
per call. Compared with the existing CUDA BF16 native d1 results, batch size
four. Inputs and candidate orders are unchanged. Each export matched the
original chosen action on **180/180 cases**.

| Set | Accuracy, all three formats | Macro F1 | Stop recall | False stops |
| --- | ---: | ---: | ---: | ---: |
| Synthetic test: 150 rows / 10 families | 90.0% | 0.893 | 30/30 | 0/90 |
| Gold: 30 authored cases | 90.0% | 0.898 | 6/6 | 1/18 |

| Export | Test max / mean absolute probability change | Gold max / mean change |
| --- | ---: | ---: |
| GGUF Q8_0 | 0.041265 / 0.000266 | 0.075710 / 0.001392 |
| MLX 8-bit | 0.037106 / 0.000351 | 0.104461 / 0.002098 |

Mean changes average over every case and all five candidates. These changes
include runtime, precision, batch shape, and quantization effects. They do not
isolate quantization alone. No threshold tuning was done. The largest changes
are large enough that downstream probability thresholds need new validation.

The original gold errors remain: useful staging validation stopped; a
deterministic parser failure retried; an expense submission with an unknown
outcome retried. Hard budgets, permissions, idempotency checks, and human
approval must remain outside the model.

These small authored sets are not real-agent safety validation. Gold is
assistant-authored, not human-adjudicated; synthetic variants are correlated.
False stops exclude true `ESCALATE` cases from their denominator. No new
validation-set evaluation, image test, 4-bit export, or production deployment
was performed.

- [GGUF metrics and predictions](../results/gguf-q8_0/)
- [MLX metrics and predictions](../results/mlx-8bit/)
- [Source, tool versions, and artifact hashes](../results/exports.json)
- [Public release commits and download verification](../results/export-releases.json)

## Reproduce conversion

Run from a clone of this repository on Apple Silicon, with uv, Git, CMake,
and a working C++ build toolchain. This is an isolated export environment;
it does not change the project's normal dependencies. Check free memory first.

```sh
uv venv --python 3.12 checkpoints/export-tools/venv
uv pip install --python checkpoints/export-tools/venv/bin/python \
  -e . -r configs/export-requirements.txt
export EXPORT_PY="$PWD/checkpoints/export-tools/venv/bin/python"
export SOURCE="$PWD/checkpoints/touch-grass-v0/merged"
checkpoints/export-tools/venv/bin/hf download heath0xFF/touch-grass-v0 \
  --revision 8cd6d51892d82b61db40f2ff4889a7dfe0755f89 --local-dir "$SOURCE"
```

Before conversion, verify the source `model.safetensors` SHA-256:
`be755f5b07ba9b00e945647c49fa7688a821a87e1ffab5011c307849bbcb7472`.
The original checkpoint's Python files are trusted code; review them first.
Use new output directories. Do not overwrite a prior export.

### GGUF Q8_0

The converter detects the source's `D1Model` marker. It writes `lfm2-d1`
metadata and the System One prompt template, rather than treating the model
as generic LFM2 chat weights.

```sh
git init checkpoints/export-tools/llama.cpp
git -C checkpoints/export-tools/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
git -C checkpoints/export-tools/llama.cpp fetch --depth 1 origin 10b8b7553494f3638da7668cf17cac089229122a
git -C checkpoints/export-tools/llama.cpp checkout --detach FETCH_HEAD
export LLAMA="$PWD/checkpoints/export-tools/llama.cpp"
cmake -S "$LLAMA" -B "$LLAMA/build" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$LLAMA/build" --target llama-server llama-quantize -j 4
mkdir checkpoints/touch-grass-v0-GGUF
"$EXPORT_PY" "$LLAMA/convert_hf_to_gguf.py" "$SOURCE" \
  --outfile checkpoints/touch-grass-v0-F16.gguf --outtype f16 --model-name touch-grass-v0
"$LLAMA/build/bin/llama-quantize" checkpoints/touch-grass-v0-F16.gguf \
  checkpoints/touch-grass-v0-GGUF/touch-grass-v0-Q8_0.gguf Q8_0 4
```

The intermediate F16 file is not published. We used the C++ quantizer after
an initial direct Python Q8 conversion emitted numeric overflow warnings.
That initial file was not released. The final file's scales and other floating
weights were checked for finite values before inference.

### MLX 8-bit

```sh
test ! -e checkpoints/touch-grass-v0-MLX-8bit && \
"$EXPORT_PY" -m mlx_vlm convert --hf-path "$SOURCE" \
  --mlx-path checkpoints/touch-grass-v0-MLX-8bit \
  -q --q-bits 8 --q-group-size 64 --dtype float16
```

This uses 8-bit affine quantization for language linear/embedding weights.
Other tensors stay float16; the vision tower and projector are not quantized.
The export's upstream PyTorch `.py` files were removed before distribution.
The `auto_map` marker stays in `config.json`: mlx-vlm uses it to select native d1.

Each release also includes `LICENSE`, `NOTICE`, source `training.json`, a
model card, `export.json`, and `request.json`. The request is
`{"state": <examples/trajectory.json>, "questions": {"action": question()}}`,
using `touch_grass.dataset.question`. These files do not change the weights.
Retain the license and record your conversion changes if you redistribute.

## Reproduce evaluation

Start the GGUF server in a separate terminal:

```sh
"$LLAMA/build/bin/llama-server" \
  -m checkpoints/touch-grass-v0-GGUF/touch-grass-v0-Q8_0.gguf \
  --host 127.0.0.1 --port 18081 --ctx-size 2048 --parallel 1 \
  --batch-size 2048 --ubatch-size 2048 --no-context-shift --no-webui -ngl 99
```

Then, from the repository root:

```sh
"$EXPORT_PY" -m touch_grass.evaluate_exports --backend gguf \
  --model http://127.0.0.1:18081 --source "$SOURCE" \
  --output checkpoints/check-gguf-q8_0
"$EXPORT_PY" -m touch_grass.evaluate_exports --backend mlx \
  --model checkpoints/touch-grass-v0-MLX-8bit --source "$SOURCE" \
  --output checkpoints/check-mlx-8bit
```

Stop the temporary server afterward. No persistent service is required.

The evaluator reuses the project's data validation, metrics, and native answer
checks. It rejects inputs over 2,048 tokens and refuses existing output paths.
For MLX it checks exact prompt token IDs and candidate token groups against the
original d1 renderer. For both formats it checks native token usage, zero
output tokens, finite probabilities, normalization, and argmax consistency.
It compares every saved prediction with the original case and candidate order.

MLX's `predict()` presentation API rounds probabilities to four decimal places.
We use its native `model.probabilities()` instead, without rounding or
renormalizing, so probability-difference measurements retain full precision.
The GGUF check uses the native endpoint's probabilities unchanged. Loop timings
include Python validation and output handling; they are not isolated latency
benchmarks.

## Upstream references

- [Liquid's d1 deployment guide](https://docs.liquid.ai/lfm/models/d1-3b)
- [Native llama.cpp d1 support](https://github.com/ggml-org/llama.cpp/pull/30110)
- [Native mlx-vlm d1 support](https://github.com/Blaizzy/mlx-vlm/pull/2467)
- [Official MLX export recipe and precision notes](https://huggingface.co/LiquidAI/d1-3B-MLX-8bit)
