from __future__ import annotations

from .local_queue import LocalEventQueue


def create_health_app(queue: LocalEventQueue):
    """Create the FastAPI app used for local health inspection.

    Args:
        queue: Local queue whose status is exposed by the API.

    Returns:
        Configured ``FastAPI`` application.
    """
    try:
        from fastapi import FastAPI
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("FastAPI is required for the health API") from exc

    app = FastAPI(title="Parking Edge Node")

    @app.get("/health")
    def health() -> dict[str, object]:
        """Return basic process and queue health.

        Returns:
            Dictionary containing overall status and queue counts.
        """
        counts = queue.counts_by_status()
        return {
            "status": "ok",
            "queue": counts,
        }

    @app.get("/metrics/queue")
    def queue_metrics() -> dict[str, int]:
        """Return outbound queue counts grouped by status.

        Returns:
            Mapping from queue status to row count.
        """
        return queue.counts_by_status()

    return app
