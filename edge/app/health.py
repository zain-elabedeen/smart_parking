from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "app"

from .api import create_health_app
from .config import load_config
from .local_queue import LocalEventQueue


def main() -> None:
    """Run the local FastAPI health server from CLI arguments.

    Input:
        Reads the config path from CLI arguments and opens the configured local
        queue database.

    Output:
        Starts a Uvicorn HTTP server exposing health and queue endpoints.
    """
    parser = argparse.ArgumentParser(description="Run parking edge health API")
    parser.add_argument("--config", default="edge/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    queue = LocalEventQueue(config.local_queue.sqlite_path)
    app = create_health_app(queue)

    import uvicorn

    uvicorn.run(app, host=config.metrics.host, port=config.metrics.port + 1)


if __name__ == "__main__":
    main()
