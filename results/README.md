# Measured v0 run

Run date: 2026-10-16. Hardware: NVIDIA DGX Spark, GB10, aarch64,
121 GiB usable unified memory, driver 580.178.04. Training and evaluation used
an isolated uv environment, not the resident inference services.

- [`base/metrics.json`](base/metrics.json): original d1-3B.
- [`finetuned/metrics.json`](finetuned/metrics.json): one fixed LoRA epoch, merged and reloaded with the native API.
- Each directory contains `validation.predictions.jsonl`, `test.predictions.jsonl`, and `gold.predictions.jsonl`.
- [`training.json`](training.json): configuration, package versions, data hashes, supervised-token check, training history, runtime, and peak Torch allocation.
- [`demo.json`](demo.json): native CUDA BF16 CLI result for `examples/trajectory.json`, with canonical candidate order and full probabilities.
- [`artifacts.json`](artifacts.json): sizes and SHA-256 hashes for the adapter, merged weights, native code, and supporting checkpoint files. All 27 files were copied back from Spark and verified.

Base and fine-tuned evaluations use the same 150 validation rows, 150 synthetic
test rows, and 30 gold rows. Predictions are in dataset order. Candidate order
comes from each stored row; evaluation batch size is 4. Both model code and
tokenizer are pinned to base revision
`da1fe36a861f24690f27f622dca1d8688503d113`.

`tests/test_saved_results.py` recalculates the metrics from these predictions,
checks input hashes and IDs, and validates the native probability distributions.
No test or gold result was used to change the learning rate, epoch count,
training labels, prompt, or checkpoint selection after the run.

The synthetic test contains only **ten independent scenario families**. Its
150 name/order variants do not support a 150-independent-case confidence claim.
The gold set has 30 assistant-authored cases, not independent human judgments.
The 100% gold stop recall is six cases; the false-stop rate is one out of 18
eligible cases. Read the failure examples before using this as an agent signal.

GPU training is seeded, but different kernels or hardware need not produce
bit-identical weights. Artifact hashes identify this run, not a guarantee that
a new run will have the same hashes.

Weights are not stored in Git. Reproduction commands are in the root README;
training writes both the adapter and the native merged checkpoint. Data files
are included in this repository. No separate Hugging Face model or dataset
release is provided.
