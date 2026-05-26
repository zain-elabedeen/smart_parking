from __future__ import annotations

from datetime import datetime, timezone

from app.direction_estimator import LineCrossingDirectionEstimator
from app.schemas import Point, TrackedVehicle


def _track(track_id: int, bbox: list[float]) -> TrackedVehicle:
    """Build a tracked vehicle fixture for direction tests.

    Args:
        track_id: Track id to assign.
        bbox: Vehicle box as ``[x1, y1, x2, y2]``.

    Returns:
        ``TrackedVehicle`` with timestamps and one centroid trajectory point.
    """
    now = datetime.now(tz=timezone.utc)
    center = Point((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return TrackedVehicle(
        track_id=track_id,
        bbox=bbox,
        class_name="car",
        confidence=0.9,
        first_seen_at=now,
        last_seen_at=now,
        trajectory=[center],
    )


def test_top_to_bottom_crossing_is_entry() -> None:
    """Verify downward crossing of a horizontal line produces ENTRY.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    estimator = LineCrossingDirectionEstimator((0, 100), (200, 100), "top_to_bottom")

    assert estimator.update(_track(1, [50, 40, 100, 80])) is None
    event = estimator.update(_track(1, [50, 120, 100, 160]))

    assert event is not None
    assert event.direction == "ENTRY"
    assert estimator.direction_for_track(1) == "ENTRY"


def test_opposite_crossing_is_exit() -> None:
    """Verify upward crossing of a horizontal entry line produces EXIT.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    estimator = LineCrossingDirectionEstimator((0, 100), (200, 100), "top_to_bottom")

    assert estimator.update(_track(7, [50, 120, 100, 160])) is None
    event = estimator.update(_track(7, [50, 40, 100, 80]))

    assert event is not None
    assert event.direction == "EXIT"


def test_left_to_right_crossing_is_entry_for_vertical_line() -> None:
    """Verify left-to-right crossing of a vertical line produces ENTRY.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    estimator = LineCrossingDirectionEstimator((100, 0), (100, 200), "left_to_right")

    assert estimator.update(_track(3, [40, 50, 80, 100])) is None
    event = estimator.update(_track(3, [120, 50, 160, 100]))

    assert event is not None
    assert event.direction == "ENTRY"
