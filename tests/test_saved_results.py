import hashlib
import json
from pathlib import Path

from touch_grass.dataset import load_rows
from touch_grass.evaluate import metrics
from touch_grass.evaluate_exports import comparison
from touch_grass.judge import validate_answer


def test_published_metrics_match_raw_predictions():
    reports = list(Path("results").glob("*/metrics.json"))
    assert reports
    for path in reports:
        report = json.loads(path.read_text())
        for split, saved in report["splits"].items():
            data_path = Path(f"data/{split}.jsonl")
            rows = load_rows(data_path)
            predictions = [
                json.loads(line)
                for line in (path.parent / f"{split}.predictions.jsonl").read_text().splitlines()
            ]
            assert len(predictions) == len(rows)
            assert saved["dataset_sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()
            for row, predicted in zip(rows, predictions, strict=True):
                assert predicted["example_id"] == row["example_id"]
                assert predicted["scenario_id"] == row["scenario_id"]
                assert predicted["candidates"] == row["candidates"]
                assert predicted["true_action"] == row["action"]
                validate_answer(
                    {"choice": predicted["action"], "probabilities": predicted["probabilities"]}
                )
                assert predicted["usage"]["output_tokens"] == 0
            computed = metrics(
                [row["action"] for row in rows], [row["action"] for row in predictions]
            )
            assert json.loads(json.dumps(computed)) == {key: saved[key] for key in computed}
            if "comparison" in saved:
                reference_path = Path(f"results/finetuned/{split}.predictions.jsonl")
                reference = [json.loads(line) for line in reference_path.read_text().splitlines()]
                assert (
                    saved["reference_sha256"]
                    == hashlib.sha256(reference_path.read_bytes()).hexdigest()
                )
                assert saved["comparison"] == comparison(reference, predictions)


def test_export_manifest_matches_evaluation():
    manifest = json.loads(Path("results/exports.json").read_text())
    source = json.loads(Path("results/huggingface.json").read_text())
    assert manifest["source"] == source
    for name, export in manifest["exports"].items():
        assert export["source_revision"] == source["revision"]
        assert export["source_weights_sha256"] == source["weights"]["sha256"]
        assert export["evaluation"] == json.loads(Path(f"results/{name}/metrics.json").read_text())
        assert {row["path"] for row in export["files"]} >= {
            "LICENSE",
            "NOTICE",
            "README.md",
            "request.json",
        }
