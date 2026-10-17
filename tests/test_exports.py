import pytest

from touch_grass import evaluate_exports as exports
from touch_grass.dataset import LABELS


def prediction(action):
    return {
        "example_id": "case-1",
        "true_action": "CONTINUE",
        "candidates": list(LABELS),
        "action": action,
        "probabilities": {label: float(label == action) for label in LABELS},
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }


def test_export_comparison():
    old = prediction("CONTINUE")
    same = exports.comparison([old], [old])
    assert same["same_top_answer"] == 1
    assert same["max_absolute_probability_change"] == 0
    changed = exports.comparison([old], [prediction("TOUCH_GRASS")])
    assert changed["same_top_answer"] == 0
    assert changed["max_absolute_probability_change"] == 1
    assert changed["mean_absolute_probability_change"] == 0.4
    assert changed["changed_answers"] == [
        {"example_id": "case-1", "from": "CONTINUE", "to": "TOUCH_GRASS"}
    ]


@pytest.mark.parametrize(
    "key,value",
    [
        ("example_id", "different"),
        ("candidates", list(reversed(LABELS))),
        ("usage", {"input_tokens": 11, "output_tokens": 0}),
    ],
)
def test_export_comparison_rejects_mismatched_inputs(key, value):
    old, new = prediction("CONTINUE"), prediction("CONTINUE")
    new[key] = value
    with pytest.raises(ValueError, match="Reference mismatch"):
        exports.comparison([old], [new])


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com:18081",
        "http://127.0.0.1:18081@example.com",
        "http://127.0.0.1:18081/path",
        "https://127.0.0.1:18081",
    ],
)
def test_export_evaluator_rejects_non_loopback_url(monkeypatch, tmp_path, url):
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_exports", "--backend", "gguf", "--model", url, "--output", str(output)],
    )
    with pytest.raises(SystemExit):
        exports.main()
    assert not output.exists()


def test_export_evaluator_refuses_existing_output(monkeypatch, tmp_path):
    marker = tmp_path / "keep.txt"
    marker.write_text("keep")
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_exports",
            "--backend",
            "gguf",
            "--model",
            "http://127.0.0.1:18081",
            "--output",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        exports.main()
    assert marker.read_text() == "keep"


def test_export_comparison_rejects_invalid_probabilities():
    old, new = prediction("CONTINUE"), prediction("CONTINUE")
    new["probabilities"]["CONTINUE"] = float("nan")
    with pytest.raises(ValueError, match="invalid candidate probabilities"):
        exports.comparison([old], [new])
