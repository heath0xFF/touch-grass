import json
from collections import Counter
from pathlib import Path

import pytest

from touch_grass.dataset import (
    LABELS,
    check_disjoint,
    load_rows,
    prompt_completion,
    question,
    state,
)
from touch_grass.generate import generate


def test_generated_data_is_reproducible_grouped_and_balanced(tmp_path):
    generate(output=tmp_path)
    splits = []
    for split, count, groups in (("train", 1200, 80), ("validation", 150, 10), ("test", 150, 10)):
        path = tmp_path / f"{split}.jsonl"
        assert path.read_bytes() == Path(f"data/{split}.jsonl").read_bytes()
        rows = load_rows(path)
        splits.append(rows)
        assert len(rows) == count
        assert len({row["scenario_id"] for row in rows}) == groups
        assert len({row["domain"] for row in rows}) == 5
        assert Counter(row["action"] for row in rows) == dict.fromkeys(LABELS, count // 5)
        for label in LABELS:
            positions = Counter(
                row["candidates"].index(label) for row in rows if row["action"] == label
            )
            assert positions == dict.fromkeys(range(5), count // 25)
        for scenario_id in {row["scenario_id"] for row in rows}:
            family = [row for row in rows if row["scenario_id"] == scenario_id]
            assert len(family) == 15
            assert len({json.dumps(state(row), sort_keys=True) for row in family}) == 15
    check_disjoint(*splits, load_rows("data/gold.jsonl"))
    assert not (tmp_path / "gold.jsonl").exists()
    assert (tmp_path / "manifest.json").read_bytes() == Path("data/manifest.json").read_bytes()


def test_gold_is_separate_and_has_explanations():
    rows = load_rows("data/gold.jsonl")
    assert len(rows) == len({row["scenario_id"] for row in rows}) == 30
    assert Counter(row["action"] for row in rows) == dict.fromkeys(LABELS, 6)
    assert all(row["rationale"] for row in rows)
    assert all(set(state(row)) == {"task", "trajectory"} for row in rows)
    all_states = [
        json.dumps(state(row), sort_keys=True) for row in rows + load_rows("data/train.jsonl")
    ]
    assert len(set(all_states)) == len(all_states)


def test_leakage_rejected():
    with pytest.raises(ValueError, match="leakage"):
        check_disjoint([{"scenario_id": "a"}], [{"scenario_id": "a"}])


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"task": "x", "trajectory": "bad"},
        {"task": "x", "trajectory": [""]},
        {"task": "", "trajectory": ["ok"]},
    ],
)
def test_bad_input_rejected(value):
    with pytest.raises(ValueError):
        state(value)


@pytest.mark.parametrize("order", [[], ["RETRY"] * 5, list(LABELS) + ["RETRY"], "RETRY", [{}] * 5])
def test_bad_candidate_order_rejected(order):
    with pytest.raises(ValueError):
        question(order)


def test_metadata_is_not_evidence():
    row = load_rows("data/gold.jsonl")[0]
    value = state(row)
    assert "action" not in value and "rationale" not in value and "domain" not in value
    assert value["trajectory"] == row["trajectory"]


def test_prompt_completion_uses_native_renderer_and_shuffled_alias():
    class Tokenizer:
        bos_token = "<bos>"
        eos_token = "<eos>"

        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return list(range(len(text)))

    class NativePrompt:
        @staticmethod
        def aliases(tokenizer, labels):
            return [(str(i), i) for i in range(len(labels))]

        @staticmethod
        def as_question(q):
            return q

        @staticmethod
        def render(tokenizer, value, q, bos):
            assert set(value) == {"task", "trajectory"}
            assert list(q["criteria"]) == list(reversed(LABELS))
            return bos + value["task"]

    row = {
        "task": "task",
        "trajectory": ["step"],
        "candidates": list(reversed(LABELS)),
        "action": "CONTINUE",
    }
    assert prompt_completion(row, Tokenizer(), NativePrompt()) == {
        "prompt": "<bos>task",
        "completion": "4<eos>",
    }
    row["task"] = "x" * 3000
    row["example_id"] = "long"
    with pytest.raises(ValueError, match="refusing truncation"):
        prompt_completion(row, Tokenizer(), NativePrompt())


def test_invalid_jsonl_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"task": "x"}\n')
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_rows(path)
