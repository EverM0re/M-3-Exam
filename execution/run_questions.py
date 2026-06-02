#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VALID_TYPES = ["SS", "MS", "TR", "TH", "II", "MR", "FM", "FJ"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate memory-evaluation questions from dialogue sessions.",
    )
    parser.add_argument(
        "--type", dest="q_type", choices=VALID_TYPES,
        metavar="{" + ",".join(VALID_TYPES) + "}",
        help="Question type (overrides config.yaml)",
    )
    parser.add_argument(
        "--num", dest="number", type=int,
        help="Total questions to generate (overrides config.yaml)",
    )
    parser.add_argument(
        "--per-turn", dest="per_turn", type=int,
        help="Questions per LLM turn (overrides config.yaml)",
    )
    parser.add_argument(
        "--rounds", dest="rounds", type=int,
        help="Consecutive dialogue rounds sampled per turn (overrides config.yaml)",
    )
    parser.add_argument(
        "--dialogue-route", dest="dialogue_route",
        help="Path to folder with sessions.json / timeline.json (overrides config.yaml)",
    )
    parser.add_argument(
        "--output-dir", dest="output_dir",
        help="Directory for question.json output (overrides config.yaml)",
    )
    return parser.parse_args()


def _apply_overrides(args: argparse.Namespace) -> None:
    from m3exam.config.config_loader import get_config

    config = get_config()
    qg_cfg = config.setdefault("question_generation", {})

    if args.q_type:
        qg_cfg["question_type"] = args.q_type
    if args.number is not None:
        qg_cfg["question_number"] = args.number
    if args.per_turn is not None:
        qg_cfg["question_per_turn"] = args.per_turn
    if args.rounds is not None:
        qg_cfg["rounds_per_sample"] = args.rounds
    if args.dialogue_route:
        qg_cfg["question_dialogue_route"] = args.dialogue_route
    if args.output_dir:
        qg_cfg["output_dir"] = args.output_dir


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    args = _parse_args()
    _apply_overrides(args)

    from m3exam.functions.question_generation.question_generator import QuestionGenerator

    generator = QuestionGenerator()

    print("=" * 60)
    print("  Question Generation Pipeline")
    print("=" * 60)
    print(f"  Type          : {generator.question_type}")
    print(f"  Target count  : {generator.question_number}")
    print(f"  Per turn      : {generator.question_per_turn}")
    print(f"  Rounds/sample : {generator.rounds_per_sample}")
    print(f"  Dialogue from : {generator.dialogue_route}")
    print(f"  Output dir    : {generator.output_dir}")
    print(f"  Sessions loaded: {len(generator.sessions)}")
    print(f"  Timeline events: {len(generator.timeline)}")
    print("=" * 60)

    questions = generator.generate()

    if questions:
        import json

        print("\nSample output (first question):")
        print(json.dumps(questions[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
