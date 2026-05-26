from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SiteConfig(BaseModel):
    """Site-level identity settings.

    Input fields:
        site_id: Parking site identifier.
        edge_device_id: Identifier for this edge node.

    Output:
        Nested section of ``AppConfig`` used when creating event payloads.
    """

    site_id: str
    edge_device_id: str


class CameraConfig(BaseModel):
    """Camera source configuration.

    Input fields:
        camera_id: Logical camera identifier.
        source: RTSP URL, file path, or webcam URI.
        width: Optional expected frame width.
        height: Optional expected frame height.

    Output:
        Nested section of ``AppConfig`` used by the video reader.
    """

    camera_id: str
    source: str
    width: int | None = None
    height: int | None = None


class FrameSamplingConfig(BaseModel):
    """Frame-rate control configuration.

    Input fields:
        target_fps: Initial frame processing rate.
        adaptive: Whether load should adjust the target rate.
        min_fps: Lower bound for adaptive sampling.
        max_fps: Upper bound for adaptive sampling.

    Output:
        Nested section of ``AppConfig`` used to create ``FrameSampler``.
    """

    target_fps: float = Field(default=8.0, gt=0)
    adaptive: bool = True
    min_fps: float = Field(default=2.0, gt=0)
    max_fps: float = Field(default=15.0, gt=0)


class DetectorConfig(BaseModel):
    """YOLO detector configuration for vehicle or plate models.

    Input fields:
        model_path: Path to a model file.
        model_url: Optional URL used by the setup script to download the model.
        confidence_threshold: Minimum confidence accepted from the detector.
        classes: Optional class allow-list, mainly used for vehicle detection.

    Output:
        Nested section of ``AppConfig`` used to create detector instances.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_path: str
    model_url: str | None = None
    confidence_threshold: float = Field(default=0.45, ge=0, le=1)
    classes: list[str] = Field(default_factory=list)


class TrackerConfig(BaseModel):
    """ByteTrack lifecycle and matching configuration.

    Input fields:
        lost_timeout_seconds: Time before an unseen track expires.
        min_track_duration_seconds: Minimum duration before a track is mature.
        track_activation_threshold: Detection confidence required to start a
            ByteTrack track.
        minimum_matching_threshold: ByteTrack association threshold.
        minimum_consecutive_frames: Number of frames required to activate a
            track.

    Output:
        Nested section of ``AppConfig`` used to create ``VehicleTracker``.
    """

    lost_timeout_seconds: float = Field(default=2.0, gt=0)
    min_track_duration_seconds: float = Field(default=0.5, ge=0)
    track_activation_threshold: float = Field(default=0.25, ge=0, le=1)
    minimum_matching_threshold: float = Field(default=0.8, ge=0, le=1)
    minimum_consecutive_frames: int = Field(default=1, ge=1)


class DirectionLineConfig(BaseModel):
    """Virtual line coordinates for direction estimation.

    Input fields:
        start: Start point as ``(x, y)``.
        end: End point as ``(x, y)``.

    Output:
        Nested section of ``DirectionConfig``.
    """

    start: tuple[float, float]
    end: tuple[float, float]


class DirectionConfig(BaseModel):
    """Direction estimation configuration.

    Input fields:
        mode: Direction strategy name, currently ``line_crossing``.
        line: Virtual line coordinates.
        entry_direction: Which crossing orientation means ``ENTRY``.

    Output:
        Nested section of ``AppConfig`` used to create the estimator.
    """

    mode: str = "line_crossing"
    line: DirectionLineConfig
    entry_direction: str = "top_to_bottom"


class KafkaConfig(BaseModel):
    """Kafka producer configuration.

    Input fields:
        bootstrap_servers: Kafka bootstrap server list.
        topic_plate_crops: Topic for plate crop events.

    Output:
        Nested section of ``AppConfig`` used by publisher setup.
    """

    bootstrap_servers: str
    topic_plate_crops: str = "plate-crops"


class LocalQueueConfig(BaseModel):
    """Local durable queue configuration.

    Input fields:
        sqlite_path: Path to the SQLite queue database.

    Output:
        Nested section of ``AppConfig`` used to create ``LocalEventQueue``.
    """

    sqlite_path: str = "data/outbound_events.db"


class ObjectStorageConfig(BaseModel):
    """Plate crop storage configuration.

    Input fields:
        mode: Storage mode; the MVP supports ``local``.
        local_path: Root directory for local crop files.

    Output:
        Nested section of ``AppConfig`` used to create ``LocalObjectStorage``.
    """

    mode: str = "local"
    local_path: str = "data/crops"


class ProcessingConfig(BaseModel):
    """Pipeline behavior controls for cost and log volume.

    Input fields:
        min_plate_track_displacement_px: Minimum track movement before running
            plate detection on that vehicle.
        plate_detection_classes: Vehicle classes eligible for plate detection.
        frame_log_interval: Log every N processed frames at INFO level.

    Output:
        Nested section of ``AppConfig`` used by the main processing loop.
    """

    min_plate_track_displacement_px: float = Field(default=20.0, ge=0)
    plate_detection_classes: list[str] = Field(default_factory=lambda: ["car", "truck", "bus"])
    frame_log_interval: int = Field(default=10, ge=1)


class MetricsConfig(BaseModel):
    """Metrics and health binding configuration.

    Input fields:
        host: Interface for metrics and health servers.
        port: Prometheus metrics port.

    Output:
        Nested section of ``AppConfig`` used by metrics and health startup.
    """

    host: str = "0.0.0.0"
    port: int = 9101


class AppConfig(BaseModel):
    """Complete edge-node configuration model.

    Input fields:
        site: Site identity settings.
        camera: Camera source settings.
        frame_sampling: Sampling behavior.
        vehicle_detector: Vehicle model settings.
        plate_detector: Plate model settings.
        tracker: Tracker behavior.
        direction: Direction estimation behavior.
        kafka: Kafka publishing settings.
        local_queue: SQLite queue settings.
        object_storage: Crop storage settings.
        processing: Runtime processing behavior.
        metrics: Metrics server settings.

    Output:
        Validated configuration object consumed by application startup.
    """

    site: SiteConfig
    camera: CameraConfig
    frame_sampling: FrameSamplingConfig
    vehicle_detector: DetectorConfig
    plate_detector: DetectorConfig
    tracker: TrackerConfig
    direction: DirectionConfig
    kafka: KafkaConfig
    local_queue: LocalQueueConfig
    object_storage: ObjectStorageConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Validated ``AppConfig`` instance.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    raw = _resolve_config_paths(raw, config_path.parent)
    return AppConfig(**raw)


def _resolve_config_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve relative filesystem paths from the config file directory.

    Args:
        raw: Unvalidated configuration dictionary.
        base_dir: Directory containing the YAML configuration file.

    Returns:
        Copy of the configuration with local path values made absolute.
    """
    resolved = deepcopy(raw)
    _resolve_nested_path(resolved, ("camera", "source"), base_dir, allow_uri=True)
    _resolve_nested_path(resolved, ("vehicle_detector", "model_path"), base_dir)
    _resolve_nested_path(resolved, ("plate_detector", "model_path"), base_dir)
    _resolve_nested_path(resolved, ("local_queue", "sqlite_path"), base_dir)
    _resolve_nested_path(resolved, ("object_storage", "local_path"), base_dir)
    return resolved


def _resolve_nested_path(raw: dict[str, Any], keys: tuple[str, ...], base_dir: Path, allow_uri: bool = False) -> None:
    """Resolve one nested path string in a mutable configuration dictionary.

    Args:
        raw: Configuration dictionary to mutate.
        keys: Nested key path to resolve.
        base_dir: Base directory for relative local paths.
        allow_uri: Whether URI schemes such as ``rtsp://`` should be preserved.

    Returns:
        ``None``.
    """
    parent: Any = raw
    for key in keys[:-1]:
        if not isinstance(parent, dict) or key not in parent:
            return
        parent = parent[key]

    final_key = keys[-1]
    value = parent.get(final_key) if isinstance(parent, dict) else None
    if not isinstance(value, str) or not value:
        return

    if allow_uri and _has_external_scheme(value):
        return

    prefix = "file://"
    if allow_uri and value.startswith(prefix):
        value = value.removeprefix(prefix)
        parent[final_key] = f"{prefix}{_resolve_path_string(value, base_dir)}"
        return

    parent[final_key] = _resolve_path_string(value, base_dir)


def _resolve_path_string(value: str, base_dir: Path) -> str:
    """Resolve one local path string against a base directory.

    Args:
        value: Local path string from configuration.
        base_dir: Base directory for relative paths.

    Returns:
        Absolute path string, or the original absolute path string.
    """
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _has_external_scheme(value: str) -> bool:
    """Check whether a path-like value is a non-file URI.

    Args:
        value: Source string from configuration.

    Returns:
        ``True`` for RTSP, HTTP, webcam, and similar external URI schemes.
    """
    parsed = urlparse(value)
    return bool(parsed.scheme and parsed.scheme != "file")
