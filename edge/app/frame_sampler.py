from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class FrameSampler:
    """Controls whether each incoming frame should be processed.

    Input fields:
        target_fps: Current target processing rate.
        adaptive: Whether latency and queue depth can adjust the rate.
        min_fps: Lower bound for adaptive adjustment.
        max_fps: Upper bound for adaptive adjustment.

    Output:
        Boolean decisions from ``should_process`` and mutable target FPS state.
    """

    target_fps: float = 8.0
    adaptive: bool = True
    min_fps: float = 2.0
    max_fps: float = 15.0
    _last_emit_at: datetime | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize the initial FPS and initialize sampling state.

        Returns:
            ``None``.
        """
        self.target_fps = min(self.max_fps, max(self.min_fps, self.target_fps))
        self._last_emit_at = None

    def should_process(self, timestamp: datetime) -> bool:
        """Decide whether a frame at this timestamp should go to inference.

        Args:
            timestamp: Timestamp of the candidate frame.

        Returns:
            ``True`` when enough time has elapsed for the target FPS,
            otherwise ``False``.
        """
        if self._last_emit_at is None:
            self._last_emit_at = timestamp
            return True
        elapsed = (timestamp - self._last_emit_at).total_seconds()
        if elapsed >= 1.0 / self.target_fps:
            self._last_emit_at = timestamp
            return True
        return False

    def adapt(self, *, inference_latency_seconds: float, queue_depth: int) -> None:
        """Adjust target FPS based on current load signals.

        Args:
            inference_latency_seconds: Latest measured inference latency.
            queue_depth: Current outbound queue depth.

        Returns:
            ``None``. Mutates ``target_fps`` within configured bounds.
        """
        if not self.adaptive:
            return
        overloaded = inference_latency_seconds > (1.0 / self.target_fps) or queue_depth > 100
        if overloaded:
            self.target_fps = max(self.min_fps, self.target_fps * 0.8)
        else:
            self.target_fps = min(self.max_fps, self.target_fps * 1.05)
