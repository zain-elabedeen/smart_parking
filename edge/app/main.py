from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "app"

from .bootstrap import RuntimeOptions, build_pipeline
from .config import load_config

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the parking edge-node CLI.

    Input:
        Reads CLI arguments and the configured YAML file.

    Output:
        Starts the edge-node processing pipeline and exits when the source ends,
        the process is interrupted, or ``--max-frames`` is reached.
    """
    args = parse_args()
    configure_logging()

    logger.info("starting parking edge node")
    logger.info("loading config path=%s", args.config)
    config = load_config(args.config)
    pipeline = build_pipeline(
        config,
        RuntimeOptions(
            source=args.source,
            no_kafka=args.no_kafka,
            verbose_tracks=args.verbose_tracks,
        ),
    )
    pipeline.run(max_frames=args.max_frames)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the edge node.

    Returns:
        Parsed ``argparse.Namespace`` with config path, source override, frame
        limit, Kafka mode, and logging verbosity options.
    """
    parser = argparse.ArgumentParser(description="Smart parking edge-node MVP")
    parser.add_argument("--config", default="edge/config.yaml")
    parser.add_argument("--source", default=None, help="Override configured camera source")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames; 0 means run forever")
    parser.add_argument("--no-kafka", action="store_true", help="Queue events locally without trying Kafka")
    parser.add_argument("--verbose-tracks", action="store_true", help="Log per-track crop and zero-plate detection details")
    return parser.parse_args()


def configure_logging() -> None:
    """Configure terminal logging for the CLI process.

    Returns:
        ``None``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )


if __name__ == "__main__":
    main()
