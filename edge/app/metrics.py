from __future__ import annotations

import logging

try:
    from prometheus_client import Counter, Gauge, start_http_server
except Exception:  # pragma: no cover - runtime dependency
    Counter = Gauge = None  # type: ignore
    start_http_server = None  # type: ignore

logger = logging.getLogger(__name__)


class EdgeMetrics:
    """Prometheus metrics registered by the edge process.

    Input:
        Created without arguments.

    Output:
        Exposes counters and gauges when ``prometheus_client`` is installed.
    """

    def __init__(self) -> None:
        """Register metric instruments if Prometheus is available.

        Returns:
            ``None``. Sets ``enabled`` to ``False`` when the dependency is
            missing.
        """
        if Counter is None or Gauge is None:
            self.enabled = False
            self.server_started = False
            return
        self.enabled = True
        self.server_started = False
        self.frames_read = Counter("edge_frames_read_total", "Frames read from camera")
        self.frames_processed = Counter("edge_frames_processed_total", "Frames sent through inference")
        self.events_enqueued = Counter("edge_events_enqueued_total", "Events enqueued for cloud publishing")
        self.inference_failures = Counter("edge_inference_failures_total", "Inference failures")
        self.queue_depth = Gauge("edge_outbound_queue_depth", "Pending outbound event count")
        self.camera_connected = Gauge("edge_camera_connected", "Camera connection status as 0 or 1")


def start_metrics_server(host: str, port: int) -> EdgeMetrics:
    """Start the Prometheus HTTP metrics server.

    Args:
        host: Interface to bind.
        port: TCP port to bind.

    Returns:
        ``EdgeMetrics`` instance with registered counters and gauges. If the
        metrics HTTP port cannot be opened, the process logs a warning and
        continues with in-process counters.
    """
    metrics = EdgeMetrics()
    if metrics.enabled and start_http_server is not None:
        try:
            start_http_server(port, addr=host)
            metrics.server_started = True
        except OSError as exc:
            logger.warning("metrics server unavailable host=%s port=%s error=%s", host, port, exc)
    return metrics
