import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

from touch_grass.dataset import BASE_MODEL, BASE_REVISION, LABELS, load_rows, write_jsonl


def metrics(truth, predictions):
    if not truth or len(truth) != len(predictions):
        raise ValueError("Expected equally sized, non-empty labels and predictions")
    if any(label not in LABELS for label in [*truth, *predictions]):
        raise ValueError("Unknown action")
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for actual, predicted in zip(truth, predictions, strict=True):
        matrix[LABELS.index(actual)][LABELS.index(predicted)] += 1
    per_class = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": support,
        }
    eligible = sum(y in LABELS[:3] for y in truth)
    false_stops = sum(
        y in LABELS[:3] and p == "TOUCH_GRASS" for y, p in zip(truth, predictions, strict=True)
    )
    return {
        "n": len(truth),
        "accuracy": sum(y == p for y, p in zip(truth, predictions, strict=True)) / len(truth),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(LABELS),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": LABELS,
            "rows": "truth",
            "columns": "prediction",
            "values": matrix,
        },
        "touch_grass_recall": per_class["TOUCH_GRASS"]["recall"],
        "false_stop_rate": false_stops / eligible if eligible else None,
        "false_stops": false_stops,
        "false_stop_denominator": eligible,
    }


def environment():
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("torch", "transformers", "touch-grass")},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


def evaluate(model_path, paths, output, device="cuda", batch_size=4):
    from touch_grass.judge import Judge

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output = Path(output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite evaluation: {output}")
    datasets = [(Path(path), load_rows(path)) for path in paths]
    if len({path.stem for path, _ in datasets}) != len(datasets):
        raise ValueError("Evaluation input filenames must have distinct stems")
    judge = Judge(model_path, device)
    output.mkdir(parents=True)
    report = {
        "model": model_path,
        "base_revision": BASE_REVISION,
        "environment": environment(),
        "splits": {},
    }
    if model_path != BASE_MODEL:
        provenance = Path(model_path) / "training.json"
        report["training"] = json.loads(provenance.read_text())
    for path, rows in datasets:
        started = time.monotonic()
        predictions = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            for row, result in zip(batch, judge.predict(batch), strict=True):
                predictions.append(
                    {
                        "example_id": row["example_id"],
                        "scenario_id": row["scenario_id"],
                        "domain": row["domain"],
                        "true_action": row["action"],
                        "candidates": row["candidates"],
                        **result,
                    }
                )
            print(f"{path.stem}: {len(predictions)}/{len(rows)}", flush=True)
        write_jsonl(output / f"{path.stem}.predictions.jsonl", predictions)
        report["splits"][path.stem] = {
            **metrics([r["action"] for r in rows], [r["action"] for r in predictions]),
            "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "seconds": time.monotonic() - started,
        }
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
