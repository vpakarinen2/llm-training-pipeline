"""Evaluation entrypoint."""

from __future__ import annotations

import argparse
import sys

from pathlib import Path
from pprint import pprint


def _add_src_to_path() -> None:
    """Ensure `src/` is on sys.path."""
    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir / "src"
    if src_dir.is_dir():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

_add_src_to_path()

from ai_pipeline.evaluation.evaluator import Evaluator  # noqa: E402
from ai_pipeline.config.loader import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Full LLM fine-tune evaluation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint directory (optional)",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation entrypoint."""
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    checkpoint_dir = Path(args.checkpoint) if args.checkpoint is not None else None

    evaluator = Evaluator(cfg, checkpoint_dir=checkpoint_dir)
    metrics = evaluator.evaluate()

    print("Evaluation metrics:")
    pprint(metrics)


if __name__ == "__main__":
    main()
