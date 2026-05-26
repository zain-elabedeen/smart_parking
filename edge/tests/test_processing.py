from __future__ import annotations

from datetime import datetime, timezone

from app.config import AppConfig
from app.processing import should_detect_plate
from app.schemas import Point, TrackedVehicle


def _config() -> AppConfig:
    """Build a minimal app config for processing-gate tests.

    Returns:
        Validated ``AppConfig`` with default processing controls.
    """
    return AppConfig(
        site={"site_id": "site-1", "edge_device_id": "edge-1"},
        camera={"camera_id": "cam-1", "source": "demo_videos/entry.mp4"},
        frame_sampling={"target_fps": 8, "adaptive": True},
        vehicle_detector={"model_path": "models/yolov8s.pt", "classes": ["car"]},
        plate_detector={"model_path": "models/license_plate_detector.pt"},
        tracker={"lost_timeout_seconds": 2.0, "min_track_duration_seconds": 0.5},
        direction={"mode": "line_crossing", "line": {"start": [0, 10], "end": [100, 10]}, "entry_direction": "top_to_bottom"},
        kafka={"bootstrap_servers": "localhost:9092", "topic_plate_crops": "plate-crops"},
        local_queue={"sqlite_path": "data/events.db"},
        object_storage={"mode": "local", "local_path": "data/crops"},
        processing={"min_plate_track_displacement_px": 20.0, "plate_detection_classes": ["car", "truck", "bus"]},
    )


def _track(class_name: str, trajectory: list[Point]) -> TrackedVehicle:
    """Build a tracked vehicle fixture.

    Args:
        class_name: Vehicle class label.
        trajectory: Track centroid path.

    Returns:
        ``TrackedVehicle`` for processing-gate tests.
    """
    now = datetime.now(tz=timezone.utc)
    return TrackedVehicle(
        track_id=1,
        bbox=[0, 0, 100, 50],
        class_name=class_name,
        confidence=0.9,
        first_seen_at=now,
        last_seen_at=now,
        trajectory=trajectory,
    )


def test_should_detect_plate_skips_disallowed_classes() -> None:
    """Verify motorcycles are skipped by the default plate-detection class gate.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    assert should_detect_plate(_track("motorcycle", [Point(0, 0), Point(100, 0)]), _config()) == (False, "class")


def test_should_detect_plate_skips_static_tracks() -> None:
    """Verify low-motion tracks do not run plate detection.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    assert should_detect_plate(_track("car", [Point(0, 0), Point(5, 0)]), _config()) == (False, "static")


def test_should_detect_plate_allows_moving_cars() -> None:
    """Verify moving allowed-class tracks run plate detection.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    assert should_detect_plate(_track("car", [Point(0, 0), Point(30, 0)]), _config()) == (True, None)
