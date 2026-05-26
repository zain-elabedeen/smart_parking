from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .schemas import Direction, Point, TrackedVehicle


EntryDirection = Literal["top_to_bottom", "bottom_to_top", "left_to_right", "right_to_left"]


@dataclass(slots=True)
class DirectionEvent:
    """A line-crossing result for one vehicle track.

    Input fields:
        track_id: Vehicle track that crossed the line.
        direction: Interpreted direction, normally ``ENTRY`` or ``EXIT``.
        crossed_at: Timestamp of the frame where crossing was detected.

    Output:
        Consumed by event-building code to decide when to publish.
    """

    track_id: int
    direction: Direction
    crossed_at: datetime


class LineCrossingDirectionEstimator:
    """Detects direction by watching track centroids cross a configured line."""

    def __init__(
        self,
        line_start: tuple[float, float],
        line_end: tuple[float, float],
        entry_direction: EntryDirection = "top_to_bottom",
        epsilon: float = 1e-6,
    ) -> None:
        """Create a line-crossing direction estimator.

        Args:
            line_start: ``(x, y)`` start coordinate of the virtual line.
            line_end: ``(x, y)`` end coordinate of the virtual line.
            entry_direction: Which crossing orientation should be labeled
                ``ENTRY``.
            epsilon: Signed-distance tolerance used to treat points as on-line.

        Returns:
            ``None``. The estimator stores per-track state internally.
        """
        self.line_start = Point(*line_start)
        self.line_end = Point(*line_end)
        self.entry_direction = entry_direction
        self.epsilon = epsilon
        self._last_side_by_track: dict[int, int] = {}
        self._direction_by_track: dict[int, Direction] = {}

    def update(self, track: TrackedVehicle) -> DirectionEvent | None:
        """Update one track and emit an event if it crossed the line.

        Args:
            track: Latest tracked vehicle state including bbox and timestamp.

        Returns:
            ``DirectionEvent`` when the track crossed since the previous update;
            otherwise ``None``.
        """
        side = self._side_sign(track.center)
        if side == 0:
            return None

        previous = self._last_side_by_track.get(track.track_id)
        self._last_side_by_track[track.track_id] = side

        if previous is None or previous == side:
            return None

        direction = self._direction_for_transition(previous, side)
        self._direction_by_track[track.track_id] = direction
        return DirectionEvent(track_id=track.track_id, direction=direction, crossed_at=track.last_seen_at)

    def direction_for_track(self, track_id: int) -> Direction:
        """Return the last known direction for a track.

        Args:
            track_id: Tracker id to look up.

        Returns:
            ``ENTRY`` or ``EXIT`` if known, otherwise ``UNKNOWN``.
        """
        return self._direction_by_track.get(track_id, "UNKNOWN")

    def forget(self, track_id: int) -> None:
        """Remove cached direction state for an expired track.

        Args:
            track_id: Tracker id to remove.

        Returns:
            ``None``.
        """
        self._last_side_by_track.pop(track_id, None)
        self._direction_by_track.pop(track_id, None)

    def _side_sign(self, point: Point) -> int:
        """Classify which side of the virtual line a point is on.

        Args:
            point: Centroid point to classify.

        Returns:
            ``-1`` or ``1`` for opposite sides of the line, or ``0`` when the
            point is within ``epsilon`` of the line.
        """
        dx = self.line_end.x - self.line_start.x
        dy = self.line_end.y - self.line_start.y
        px = point.x - self.line_start.x
        py = point.y - self.line_start.y
        signed = dx * py - dy * px
        if abs(signed) <= self.epsilon:
            return 0
        return 1 if signed > 0 else -1

    def _direction_for_transition(self, previous: int, current: int) -> Direction:
        """Map a side transition to ``ENTRY`` or ``EXIT``.

        Args:
            previous: Previous side sign from ``_side_sign``.
            current: Current side sign from ``_side_sign``.

        Returns:
            ``ENTRY`` when the transition matches the configured entry
            orientation; otherwise ``EXIT``.
        """
        entry_transition = {
            "top_to_bottom": (-1, 1),
            "bottom_to_top": (1, -1),
            "left_to_right": (1, -1),
            "right_to_left": (-1, 1),
        }[self.entry_direction]
        return "ENTRY" if (previous, current) == entry_transition else "EXIT"
