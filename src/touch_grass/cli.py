import argparse
import json
from pathlib import Path

from touch_grass.dataset import BASE_MODEL, state


def main(argv=None):
    parser = argparse.ArgumentParser(prog="touch-grass")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser(
        "generate", help="Build grouped synthetic splits; never writes gold"
    )
    generate.add_argument("--scenarios", default="data/scenarios.json")
    generate.add_argument("--output", default="data")
    generate.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train", help="Run Liquid's native answer-slot LoRA recipe")
    train.add_argument("--config", default="configs/spark.json")
    evaluate = commands.add_parser(
        "evaluate", help="Save metrics and native candidate probabilities"
    )
    evaluate.add_argument("--data", nargs="+", default=["data/test.jsonl", "data/gold.jsonl"])
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--batch-size", type=int, default=4)
    judge = commands.add_parser("judge", help="Judge one trajectory; does not execute the action")
    judge.add_argument("input", type=Path)
    judge.add_argument("--json", action="store_true")
    for command in (evaluate, judge):
        command.add_argument(
            "--model",
            default=BASE_MODEL if command is evaluate else "checkpoints/touch-grass-v0/merged",
            help="LiquidAI/d1-3B or a merged checkpoint directory",
        )
        command.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            from touch_grass.generate import generate

            generate(args.scenarios, args.output, args.seed)
        elif args.command == "train":
            from touch_grass.train import train

            train(args.config)
        elif args.command == "evaluate":
            from touch_grass.evaluate import evaluate

            evaluate(args.model, args.data, args.output, args.device, args.batch_size)
        else:
            from touch_grass.judge import Judge

            example = json.loads(args.input.read_text())
            state(example)
            result = Judge(args.model, args.device).predict([example])[0]
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                for action, probability in sorted(
                    result["probabilities"].items(), key=lambda item: item[1], reverse=True
                ):
                    print(f"{action:17} {probability:.6f}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"touch-grass: {error}\n")
    except ImportError as error:
        parser.exit(2, f"touch-grass: {error}; install with uv sync --extra train\n")


if __name__ == "__main__":
    main()
