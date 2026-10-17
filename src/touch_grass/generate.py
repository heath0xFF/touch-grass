import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from touch_grass.dataset import LABELS, check_disjoint, state, write_jsonl

PROJECTS = (
    "Alder",
    "Birch",
    "Cedar",
    "Dune",
    "Elm",
    "Fjord",
    "Grove",
    "Harbor",
    "Iris",
    "Juniper",
    "Kestrel",
    "Larch",
    "Maple",
    "North",
    "Orchard",
)


def generate(scenarios="data/scenarios.json", output="data", seed=42):
    source = Path(scenarios)
    families = json.loads(source.read_text())
    if len(families) != 100 or len({s["scenario_id"] for s in families}) != 100:
        raise ValueError("v0 expects 100 distinct scenario families")
    domains = sorted({s["domain"] for s in families})
    if len(domains) != 5:
        raise ValueError("v0 expects five domains")
    rng = random.Random(seed)
    splits = {"train": [], "validation": [], "test": []}
    rng.shuffle(domains)
    for label_index, label in enumerate(LABELS):
        for domain_index, domain in enumerate(domains):
            group = sorted(
                (s for s in families if s["action"] == label and s["domain"] == domain),
                key=lambda s: s["scenario_id"],
            )
            if len(group) != 4:
                raise ValueError(f"Expected four families for {domain}/{label}")
            rng.shuffle(group)
            for index, scenario in enumerate(group):
                split = "train"
                if index == 0 and domain_index in (label_index, (label_index + 1) % 5):
                    split = "validation"
                elif index == 0 and domain_index in ((label_index + 2) % 5, (label_index + 3) % 5):
                    split = "test"
                positions = list(range(5)) * 3
                rng.shuffle(positions)
                for variant, project in enumerate(PROJECTS):
                    order = [name for name in LABELS if name != label]
                    rng.shuffle(order)
                    order.insert(positions[variant], label)
                    row = {
                        "example_id": f"{scenario['scenario_id']}-{variant:02d}",
                        "scenario_id": scenario["scenario_id"],
                        "domain": domain,
                        "task": scenario["task"].format(project=project),
                        "trajectory": [
                            step.format(project=project) for step in scenario["trajectory"]
                        ],
                        "action": label,
                        "candidates": order,
                    }
                    state(row)
                    splits[split].append(row)
    check_disjoint(*splits.values())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "variants_per_scenario": len(PROJECTS),
        "method": "Authored synthetic templates; project-name substitutions and candidate permutations. Not independent trajectories.",
        "splits": {},
    }
    for split, rows in splits.items():
        rng.shuffle(rows)
        path = output / f"{split}.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][split] = {
            "examples": len(rows),
            "scenarios": len({r["scenario_id"] for r in rows}),
            "labels": dict(Counter(r["action"] for r in rows)),
            "domains": dict(Counter(r["domain"] for r in rows)),
            "correct_position_by_label": {
                label: [
                    sum(r["action"] == label and r["candidates"].index(label) == i for r in rows)
                    for i in range(5)
                ]
                for label in LABELS
            },
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
