"""Check native GGUF/MLX decisions against the published CUDA BF16 results."""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

from touch_grass.dataset import LABELS, MAX_TOKENS, load_rows, question, state, write_jsonl
from touch_grass.evaluate import metrics
from touch_grass.judge import validate_answer


def comparison(reference, predictions):
    if len(reference) != len(predictions) or not reference:
        raise ValueError("Expected equal, non-empty prediction sets")
    deltas, changes = [], []
    for old, new in zip(reference, predictions, strict=True):
        for key in ("example_id", "true_action", "candidates", "usage"):
            if old[key] != new[key]:
                raise ValueError(f"Reference mismatch: {key} for {new['example_id']}")
        for row in (old, new):
            validate_answer({"choice": row["action"], "probabilities": row["probabilities"]})
        deltas.extend(abs(old["probabilities"][k] - new["probabilities"][k]) for k in LABELS)
        if old["action"] != new["action"]:
            changes.append(
                {"example_id": new["example_id"], "from": old["action"], "to": new["action"]}
            )
    return {
        "reference": "Published CUDA BF16 native d1, batch size 4",
        "note": "Changes include runtime, precision, batch shape, and quantization effects.",
        "same_top_answer": len(reference) - len(changes),
        "max_absolute_probability_change": max(deltas),
        "mean_absolute_probability_change": sum(deltas) / len(deltas),
        "changed_answers": changes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("gguf", "mlx"), required=True)
    parser.add_argument(
        "--model", required=True, help="Loopback llama-server URL, or local MLX directory"
    )
    parser.add_argument("--source", type=Path, default=Path("checkpoints/touch-grass-v0/merged"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an evaluation directory")
    if args.backend == "gguf":
        url = urlsplit(args.model)
        if (
            url.scheme != "http"
            or url.hostname != "127.0.0.1"
            or url.username
            or url.password
            or url.path not in ("", "/")
            or url.query
            or url.fragment
        ):
            parser.error("Use a loopback llama-server URL")

    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location("native_d1_prompt", args.source / "prompt.py")
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    tokenizer = AutoTokenizer.from_pretrained(args.source)
    if args.backend == "mlx":
        from mlx_vlm import load
        from mlx_vlm.models.d1 import prompt as mlx_prompt

        model, processor = load(args.model)
    args.output.mkdir(parents=True)
    report = {
        "backend": args.backend,
        "model": args.model,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "transformers": version("transformers"),
        "batch_size": 1,
        "splits": {},
    }
    if args.backend == "mlx":
        report["packages"] = {name: version(name) for name in ("mlx", "mlx-vlm", "numpy")}
        report["readout"] = "Native model.probabilities; predict() rounds to four decimal places"
    for split in ("test", "gold"):
        path = Path("data") / f"{split}.jsonl"
        rows = load_rows(path)
        predictions = []
        started = time.monotonic()
        for row in rows:
            value, q = state(row), question(row["candidates"])
            typed = native.as_question(q)
            text = native.render(tokenizer, value, typed, bos=tokenizer.bos_token or "")
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > MAX_TOKENS:
                raise ValueError("Input exceeds the 2,048-token v0 limit; no truncation")
            if args.backend == "mlx":
                trunk, branches, vision = model.decision_inputs(processor, value, [q])
                if trunk != ids or branches or vision:
                    raise ValueError("MLX input differs from the native d1 prompt")
                if mlx_prompt.readout_ids(processor.tokenizer, q) != native.readout_ids(
                    tokenizer, typed
                ):
                    raise ValueError("MLX candidate token groups differ from native d1")
                # Keep native probabilities before the presentation API rounds them.
                probabilities, count = model.probabilities(processor, value, [q])
                probs = dict(zip(q["criteria"], probabilities[0], strict=True))
                answer = {"choice": max(probs, key=probs.get), "probabilities": probs}
                usage = {"input_tokens": count, "output_tokens": 0}
            else:
                request = urllib.request.Request(
                    args.model.rstrip("/") + "/v1/systemone",
                    data=json.dumps({"state": value, "questions": {"action": q}}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                answer, usage = result["answers"]["action"], result["usage"]
            if usage != {"input_tokens": len(ids), "output_tokens": 0}:
                raise ValueError(f"Native token counts differ: {row['example_id']}")
            predictions.append(
                {
                    "example_id": row["example_id"],
                    "scenario_id": row["scenario_id"],
                    "domain": row["domain"],
                    "true_action": row["action"],
                    "candidates": row["candidates"],
                    **validate_answer(answer),
                    "usage": usage,
                }
            )
            if len(predictions) % 10 == 0:
                print(f"{split}: {len(predictions)}/{len(rows)}", flush=True)
        reference_path = Path("results/finetuned") / f"{split}.predictions.jsonl"
        reference = [json.loads(line) for line in reference_path.read_text().splitlines()]
        report["splits"][split] = {
            **metrics([r["action"] for r in rows], [p["action"] for p in predictions]),
            "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "seconds": time.monotonic() - started,
            "comparison": comparison(reference, predictions),
        }
        write_jsonl(args.output / f"{split}.predictions.jsonl", predictions)
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
