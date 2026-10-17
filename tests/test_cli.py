import json

import pytest

from touch_grass.cli import main
from touch_grass.dataset import LABELS


def test_judge_text_and_json_use_native_values(tmp_path, monkeypatch, capsys):
    probs = dict(zip(LABELS, [0.01, 0.01, 0.07, 0.03, 0.88], strict=True))

    class Judge:
        def __init__(self, model, device):
            assert model == "checkpoints/touch-grass-v0/merged"

        def predict(self, examples):
            assert examples == [{"task": "Check", "trajectory": ["Done"]}]
            return [
                {"action": "TOUCH_GRASS", "probabilities": probs, "usage": {"output_tokens": 0}}
            ]

    monkeypatch.setattr("touch_grass.judge.Judge", Judge)
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"task": "Check", "trajectory": ["Done"]}))
    main(["judge", str(path)])
    assert capsys.readouterr().out.splitlines()[0].split() == ["TOUCH_GRASS", "0.880000"]
    main(["judge", str(path), "--json"])
    assert json.loads(capsys.readouterr().out)["probabilities"] == probs


def test_bad_input_does_not_load_model(tmp_path, monkeypatch, capsys):
    def fail(*args):
        pytest.fail("Invalid input must not load the model")

    monkeypatch.setattr("touch_grass.judge.Judge", fail)
    path = tmp_path / "input.json"
    path.write_text('{"task":"Check", "trajectory":[]}')
    with pytest.raises(SystemExit) as error:
        main(["judge", str(path)])
    assert error.value.code == 2
    assert "trajectory" in capsys.readouterr().err
