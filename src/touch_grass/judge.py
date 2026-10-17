import importlib
import math
from pathlib import Path

from touch_grass.dataset import BASE_MODEL, BASE_REVISION, LABELS, MAX_TOKENS, question, state


def checkpoint_path(model_path):
    if model_path == BASE_MODEL:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION))
    path = Path(model_path)
    if not path.is_dir():
        raise ValueError(f"Model directory does not exist: {path}")
    return path


def validate_answer(answer):
    probs = answer["probabilities"]
    if set(probs) != set(LABELS) or any(
        not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1
        for p in probs.values()
    ):
        raise ValueError("d1 returned invalid candidate probabilities")
    if not math.isclose(sum(probs.values()), 1, abs_tol=1e-5):
        raise ValueError("d1 candidate probabilities do not sum to one")
    if answer["choice"] not in probs or probs[answer["choice"]] != max(probs.values()):
        raise ValueError("d1 choice does not match its probabilities")
    return {"action": answer["choice"], "probabilities": probs}


class Judge:
    def __init__(self, model_path=BASE_MODEL, device="cuda"):
        import torch
        from transformers import AutoModel

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable; install a CUDA PyTorch build or use --device cpu"
            )
        path = checkpoint_path(model_path)
        self.model = (
            AutoModel.from_pretrained(
                path,
                trust_remote_code=True,
                dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            )
            .to(device)
            .eval()
        )
        self.native_prompt = importlib.import_module(
            self.model.__class__.__module__.rsplit(".", 1)[0] + ".prompt"
        )

    def predict(self, examples):
        requests = []
        for example in examples:
            value = state(example)
            q = question(example.get("candidates", LABELS))
            length = self.model.engine.tokens(value, [self.native_prompt.as_question(q)])
            if length > MAX_TOKENS:
                raise ValueError(
                    f"Input has {length} tokens; v0 limit is {MAX_TOKENS}; no truncation"
                )
            requests.append((value, {"action": q}))
        responses = self.model.system_one_batch(requests)
        if len(responses) != len(examples):
            raise ValueError("d1 returned an unexpected number of decisions")
        return [
            {**validate_answer(response["answers"]["action"]), "usage": response["usage"]}
            for response in responses
        ]
