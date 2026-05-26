from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import AppConfig
from .crop_selector import PlateCropSelector
from .direction_estimator import LineCrossingDirectionEstimator
from .event_service import PlateCropEventPublisher
from .frame_sampler import FrameSampler
from .local_queue import LocalEventQueue
from .metrics import start_metrics_server
from .object_storage import LocalObjectStorage
from .pipeline import EdgePipeline
from .plate_detector import PlateDetector
from .publisher import KafkaProducerClient, QueueBackedPublisher
from .vehicle_detector import VehicleDetector
from .vehicle_tracker import VehicleTracker
from .video_reader import VideoReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Runtime options that come from CLI arguments.

    Input fields:
        source: Optional source override; ``None`` uses config.
        no_kafka: Whether Kafka publishing should be disabled.
        verbose_tracks: Whether detailed per-track logs should be enabled.

    Output:
        Passed to bootstrap code when constructing the runtime pipeline.
    """

    source: str | None
    no_kafka: bool
    verbose_tracks: bool


def build_pipeline(config: AppConfig, options: RuntimeOptions) -> EdgePipeline:
    """Construct and wire all runtime components.

    Args:
        config: Validated application configuration.
        options: Runtime options from the CLI.

    Returns:
        Fully initialized ``EdgePipeline`` ready to run.
    """
    source = options.source or config.camera.source
    _log_config_summary(config, source, options)

    sampler = _build_sampler(config)
    vehicle_detector = _build_vehicle_detector(config)
    plate_detector = _build_plate_detector(config)
    tracker = _build_tracker(config)
    direction = _build_direction_estimator(config)
    queue = _build_local_queue(config)
    metrics = _build_metrics(config)
    storage = _build_storage(config)
    publisher = _build_publisher(config, queue, options.no_kafka)
    event_publisher = PlateCropEventPublisher(config=config, storage=storage, publisher=publisher, queue=queue, metrics=metrics)
    reader = _build_reader(config, source)

    logger.info("startup complete; entering processing loop")
    return EdgePipeline(
        config=config,
        reader=reader,
        sampler=sampler,
        vehicle_detector=vehicle_detector,
        tracker=tracker,
        plate_detector=plate_detector,
        direction=direction,
        selector=PlateCropSelector(),
        event_publisher=event_publisher,
        metrics=metrics,
        verbose_tracks=options.verbose_tracks,
    )


def _log_config_summary(config: AppConfig, source: str, options: RuntimeOptions) -> None:
    """Log the high-level runtime configuration.

    Args:
        config: Validated application configuration.
        source: Resolved video source.
        options: Runtime options from the CLI.

    Returns:
        ``None``.
    """
    logger.info(
        "config loaded site_id=%s camera_id=%s edge_device_id=%s source=%s no_kafka=%s",
        config.site.site_id,
        config.camera.camera_id,
        config.site.edge_device_id,
        source,
        options.no_kafka,
    )
    logger.info(
        "processing controls min_plate_track_displacement_px=%.1f plate_detection_classes=%s frame_log_interval=%s verbose_tracks=%s",
        config.processing.min_plate_track_displacement_px,
        ",".join(config.processing.plate_detection_classes),
        config.processing.frame_log_interval,
        options.verbose_tracks,
    )


def _build_sampler(config: AppConfig) -> FrameSampler:
    """Create the frame sampler.

    Args:
        config: Validated application configuration.

    Returns:
        Configured ``FrameSampler``.
    """
    logger.info(
        "starting frame sampler target_fps=%.2f adaptive=%s min_fps=%.2f max_fps=%.2f",
        config.frame_sampling.target_fps,
        config.frame_sampling.adaptive,
        config.frame_sampling.min_fps,
        config.frame_sampling.max_fps,
    )
    return FrameSampler(**config.frame_sampling.model_dump())


def _build_vehicle_detector(config: AppConfig) -> VehicleDetector:
    """Create the vehicle detector.

    Args:
        config: Validated application configuration.

    Returns:
        Loaded ``VehicleDetector``.
    """
    logger.info(
        "loading vehicle detector model=%s confidence_threshold=%.2f classes=%s",
        config.vehicle_detector.model_path,
        config.vehicle_detector.confidence_threshold,
        ",".join(config.vehicle_detector.classes),
    )
    detector = VehicleDetector(
        config.vehicle_detector.model_path,
        config.vehicle_detector.confidence_threshold,
        config.vehicle_detector.classes,
    )
    logger.info("vehicle detector loaded successfully")
    return detector


def _build_plate_detector(config: AppConfig) -> PlateDetector:
    """Create the license plate detector.

    Args:
        config: Validated application configuration.

    Returns:
        Loaded ``PlateDetector``.
    """
    logger.info(
        "loading plate detector model=%s confidence_threshold=%.2f",
        config.plate_detector.model_path,
        config.plate_detector.confidence_threshold,
    )
    detector = PlateDetector(
        config.plate_detector.model_path,
        config.plate_detector.confidence_threshold,
    )
    logger.info("plate detector loaded successfully")
    return detector


def _build_tracker(config: AppConfig) -> VehicleTracker:
    """Create the ByteTrack-backed vehicle tracker.

    Args:
        config: Validated application configuration.

    Returns:
        Configured ``VehicleTracker``.
    """
    logger.info(
        "starting ByteTrack lost_timeout_seconds=%.2f min_track_duration_seconds=%.2f activation_threshold=%.2f matching_threshold=%.2f",
        config.tracker.lost_timeout_seconds,
        config.tracker.min_track_duration_seconds,
        config.tracker.track_activation_threshold,
        config.tracker.minimum_matching_threshold,
    )
    return VehicleTracker(**config.tracker.model_dump(), frame_rate=config.frame_sampling.target_fps)


def _build_direction_estimator(config: AppConfig) -> LineCrossingDirectionEstimator:
    """Create the line-crossing direction estimator.

    Args:
        config: Validated application configuration.

    Returns:
        Configured ``LineCrossingDirectionEstimator``.
    """
    logger.info(
        "direction estimator configured mode=%s line_start=%s line_end=%s entry_direction=%s",
        config.direction.mode,
        config.direction.line.start,
        config.direction.line.end,
        config.direction.entry_direction,
    )
    return LineCrossingDirectionEstimator(
        tuple(config.direction.line.start),
        tuple(config.direction.line.end),
        config.direction.entry_direction,  # type: ignore[arg-type]
    )


def _build_local_queue(config: AppConfig) -> LocalEventQueue:
    """Create the durable local event queue.

    Args:
        config: Validated application configuration.

    Returns:
        Open ``LocalEventQueue``.
    """
    logger.info("opening local queue sqlite_path=%s", config.local_queue.sqlite_path)
    return LocalEventQueue(config.local_queue.sqlite_path)


def _build_metrics(config: AppConfig):
    """Create and start the metrics registry/server.

    Args:
        config: Validated application configuration.

    Returns:
        ``EdgeMetrics`` instance.
    """
    logger.info("starting metrics server host=%s port=%s", config.metrics.host, config.metrics.port)
    return start_metrics_server(config.metrics.host, config.metrics.port)


def _build_storage(config: AppConfig) -> LocalObjectStorage:
    """Create crop object storage.

    Args:
        config: Validated application configuration.

    Returns:
        ``LocalObjectStorage`` rooted at the configured path.
    """
    logger.info("initializing crop storage mode=%s path=%s", config.object_storage.mode, config.object_storage.local_path)
    return LocalObjectStorage(config.object_storage.local_path)


def _build_publisher(config: AppConfig, queue: LocalEventQueue, no_kafka: bool) -> QueueBackedPublisher:
    """Create the queue-backed event publisher.

    Args:
        config: Validated application configuration.
        queue: Durable local event queue.
        no_kafka: Whether Kafka publishing should be disabled.

    Returns:
        ``QueueBackedPublisher`` using Kafka when enabled.
    """
    if no_kafka:
        logger.info("Kafka disabled by --no-kafka; events will stay in local queue")
        producer = None
    else:
        logger.info("initializing Kafka producer bootstrap_servers=%s topic=%s", config.kafka.bootstrap_servers, config.kafka.topic_plate_crops)
        producer = KafkaProducerClient(config.kafka.bootstrap_servers)
    return QueueBackedPublisher(queue=queue, producer=producer)


def _build_reader(config: AppConfig, source: str) -> VideoReader:
    """Create the video reader.

    Args:
        config: Validated application configuration.
        source: Resolved video source.

    Returns:
        Configured ``VideoReader``.
    """
    logger.info("opening video source=%s", source)
    return VideoReader(source, site_id=config.site.site_id, camera_id=config.camera.camera_id)
