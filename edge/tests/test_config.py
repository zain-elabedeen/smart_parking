from __future__ import annotations

from pathlib import Path

from app.config import load_config


def test_load_config_resolves_relative_paths_from_config_file(tmp_path) -> None:
    """Verify local paths are resolved relative to the YAML file directory.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
site:
  site_id: test-site
  edge_device_id: edge-1
camera:
  camera_id: cam-1
  source: demo_videos/input.mp4
frame_sampling:
  target_fps: 8
  adaptive: true
vehicle_detector:
  model_path: models/yolov8s.pt
  confidence_threshold: 0.45
  classes: [car]
plate_detector:
  model_path: models/license_plate_detector.pt
  confidence_threshold: 0.4
tracker:
  lost_timeout_seconds: 2.0
  min_track_duration_seconds: 0.5
direction:
  mode: line_crossing
  line:
    start: [0, 10]
    end: [100, 10]
  entry_direction: top_to_bottom
kafka:
  bootstrap_servers: localhost:9092
  topic_plate_crops: plate-crops
local_queue:
  sqlite_path: data/events.db
object_storage:
  mode: local
  local_path: data/crops
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert Path(config.camera.source) == tmp_path / "demo_videos/input.mp4"
    assert Path(config.vehicle_detector.model_path) == tmp_path / "models/yolov8s.pt"
    assert Path(config.plate_detector.model_path) == tmp_path / "models/license_plate_detector.pt"
    assert Path(config.local_queue.sqlite_path) == tmp_path / "data/events.db"
    assert Path(config.object_storage.local_path) == tmp_path / "data/crops"


def test_load_config_preserves_external_camera_uri(tmp_path) -> None:
    """Verify RTSP and webcam-style sources are not treated as local paths.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
site:
  site_id: test-site
  edge_device_id: edge-1
camera:
  camera_id: cam-1
  source: rtsp://camera/live
frame_sampling:
  target_fps: 8
  adaptive: true
vehicle_detector:
  model_path: /models/yolov8s.pt
  confidence_threshold: 0.45
  classes: [car]
plate_detector:
  model_path: /models/license_plate_detector.pt
  confidence_threshold: 0.4
tracker:
  lost_timeout_seconds: 2.0
  min_track_duration_seconds: 0.5
direction:
  mode: line_crossing
  line:
    start: [0, 10]
    end: [100, 10]
  entry_direction: top_to_bottom
kafka:
  bootstrap_servers: localhost:9092
  topic_plate_crops: plate-crops
local_queue:
  sqlite_path: /data/events.db
object_storage:
  mode: local
  local_path: /data/crops
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.camera.source == "rtsp://camera/live"

