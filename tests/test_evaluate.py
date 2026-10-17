import pytest

from touch_grass.dataset import LABELS
from touch_grass.evaluate import metrics
from touch_grass.judge import validate_answer


def test_metrics_known_confusion_and_false_stop_denominator():
    truth = [*LABELS, "CONTINUE", "ESCALATE"]
    predictions = [
        "CONTINUE",
        "TOUCH_GRASS",
        "CHANGE_STRATEGY",
        "TOUCH_GRASS",
        "ESCALATE",
        "RETRY",
        "ESCALATE",
    ]
    result = metrics(truth, predictions)
    assert result["accuracy"] == 3 / 7
    assert result["false_stops"] == 1
    assert result["false_stop_denominator"] == 4
    assert result["false_stop_rate"] == 1 / 4
    assert result["touch_grass_recall"] == 0
    assert result["per_class"]["CONTINUE"] == {
        "precision": 1,
        "recall": 0.5,
        "f1": 2 / 3,
        "support": 2,
    }
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0 + 1 + 0.5 + 0) / 5)
    assert result["confusion_matrix"]["values"] == [
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0],
    ]


def test_perfect_and_undefined_metrics():
    result = metrics(list(LABELS), list(LABELS))
    assert result["accuracy"] == result["macro_f1"] == result["touch_grass_recall"] == 1
    assert result["false_stop_rate"] == 0
    assert metrics(["ESCALATE"], ["TOUCH_GRASS"])["false_stop_rate"] is None


@pytest.mark.parametrize("truth,predicted", [([], []), (["RETRY"], []), (["unknown"], ["RETRY"])])
def test_invalid_metrics_inputs(truth, predicted):
    with pytest.raises(ValueError):
        metrics(truth, predicted)


def test_native_probabilities_are_unchanged():
    probs = {label: p for label, p in zip(LABELS, [0.03, 0.02, 0.1, 0.05, 0.8], strict=True)}
    answer = validate_answer({"choice": "TOUCH_GRASS", "probabilities": probs, "confidence": 0.8})
    assert answer["probabilities"] is probs
    assert set(answer) == {"action", "probabilities"}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 1.1, "0.2"])
def test_bad_probabilities_rejected(bad):
    probs = dict.fromkeys(LABELS, 0.2)
    probs["RETRY"] = bad
    with pytest.raises(ValueError):
        validate_answer({"choice": "CONTINUE", "probabilities": probs})


def test_probability_sum_and_choice_checked():
    with pytest.raises(ValueError, match="sum"):
        validate_answer({"choice": "CONTINUE", "probabilities": dict.fromkeys(LABELS, 0.1)})
    with pytest.raises(ValueError, match="choice"):
        validate_answer(
            {
                "choice": "CONTINUE",
                "probabilities": dict(zip(LABELS, [0.1, 0.6, 0.1, 0.1, 0.1], strict=True)),
            }
        )
