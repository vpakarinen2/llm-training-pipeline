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
from ai_pipeline.utils.logging import get_logger  # noqa: E402
from ai_pipeline.utils.seed import set_seed  # noqa: E402


logger = get_logger(__name__)


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

    logger.info("Loaded config from %s", config_path)

    set_seed(cfg.run.seed, cfg.run.deterministic)

    checkpoint_dir = Path(args.checkpoint) if args.checkpoint is not None else None

    if checkpoint_dir is not None:
        logger.info("Evaluating checkpoint at %s", checkpoint_dir)
    else:
        logger.info("Evaluating base model %s", cfg.model.model_name)

    evaluator = Evaluator(cfg, checkpoint_dir=checkpoint_dir)
    metrics = evaluator.evaluate()

    logger.info("Evaluation metrics:")
    pprint(metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception in evaluation script")
        sys.exit(1)
