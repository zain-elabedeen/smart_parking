from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .config import AppConfig
from .crop_selector import PlateCropCandidate
from .local_queue import LocalEventQueue
from .object_storage import LocalObjectStorage
from .publisher import QueueBackedPublisher
from .schemas import DebugPayload, Direction, Frame, PlateCropDetected, PlatePayload, TrackedVehicle, VehiclePayload, stable_event_id

logger = logging.getLogger(__name__)


class PlateCropEventPublisher:
    """Builds, stores, queues, and flushes plate-crop events.

    Input:
        Created with app configuration, crop storage, durable queue, publisher,
        and metrics objects.

    Output:
        Saves selected crop images and enqueues ``PlateCropDetected`` JSON
        events for eventual cloud publishing.
    """

    def __init__(
        self,
        *,
        config: AppConfig,
        storage: LocalObjectStorage,
        publisher: QueueBackedPublisher,
        queue: LocalEventQueue,
        metrics: Any,
    ) -> None:
        """Create a plate-crop event publishing service.

        Args:
            config: Validated application configuration.
            storage: Object storage implementation for crop persistence.
            publisher: Queue-backed publisher facade.
            queue: Local queue used for queue-depth metrics.
            metrics: Metrics registry object.

        Returns:
            ``None``.
        """
        self.config = config
        self.storage = storage
        self.publisher = publisher
        self.queue = queue
        self.metrics = metrics

    def publish(
        self,
        *,
        frame: Frame,
        track: TrackedVehicle,
        direction_value: Direction,
        event_timestamp: datetime,
        candidate: PlateCropCandidate,
    ) -> bool:
        """Persist a crop and enqueue a ``PlateCropDetected`` event.

        Args:
            frame: Current frame used for debug dimensions.
            track: Vehicle track being published.
            direction_value: Direction attached to the event.
            event_timestamp: Event timestamp.
            candidate: Selected plate crop candidate.

        Returns:
            ``True`` when the event was enqueued or already existed, otherwise
            ``False``.
        """
        event = self._build_event(
            frame=frame,
            track=track,
            direction_value=direction_value,
            event_timestamp=event_timestamp,
            candidate=candidate,
        )
        logger.info(
            "publishing candidate track=%s direction=%s event_timestamp=%s crop_key=%s quality=%.3f",
            track.track_id,
            direction_value,
            event_timestamp.isoformat(),
            event.plate.crop_object_key,
            candidate.quality_score,
        )
        self.storage.save_jpeg(event.plate.crop_object_key, candidate.crop)
        candidate.object_key = event.plate.crop_object_key
        logger.info(
            "saved crop track=%s object_key=%s crop_shape=%s",
            track.track_id,
            event.plate.crop_object_key,
            getattr(candidate.crop, "shape", None),
        )

        inserted = self.publisher.enqueue(event_id=event.event_id, topic=self.config.kafka.topic_plate_crops, payload_json=event.to_json())
        logger.info("queued event_id=%s topic=%s inserted=%s", event.event_id, self.config.kafka.topic_plate_crops, inserted)
        sent = self.publisher.flush_pending(key=f"{self.config.site.site_id}:{self.config.camera.camera_id}")
        queue_counts = self.queue.counts_by_status()
        logger.info("queue flush complete sent=%s queue_counts=%s", sent, queue_counts)

        if self.metrics.enabled:
            self.metrics.events_enqueued.inc()
            self.metrics.queue_depth.set(queue_counts.get("PENDING", 0))

        logger.info("enqueued %s track=%s direction=%s", event.event_id, event.track_id, event.direction)
        return True

    def _build_event(
        self,
        *,
        frame: Frame,
        track: TrackedVehicle,
        direction_value: Direction,
        event_timestamp: datetime,
        candidate: PlateCropCandidate,
    ) -> PlateCropDetected:
        """Create the outgoing event model for one selected crop.

        Args:
            frame: Current frame used for debug dimensions.
            track: Vehicle track being published.
            direction_value: Direction attached to the event.
            event_timestamp: Event timestamp.
            candidate: Selected plate crop candidate.

        Returns:
            Validated ``PlateCropDetected`` event model.
        """
        event_id = stable_event_id(self.config.site.site_id, self.config.camera.camera_id, track.track_id, event_timestamp.isoformat())
        crop_key = f"plate-crops/{self.config.site.site_id}/{self.config.camera.camera_id}/{event_id}.jpg"
        height, width = frame.image.shape[:2]
        return PlateCropDetected(
            event_id=event_id,
            site_id=self.config.site.site_id,
            camera_id=self.config.camera.camera_id,
            edge_device_id=self.config.site.edge_device_id,
            track_id=track.track_id,
            timestamp=event_timestamp,
            direction=direction_value,
            vehicle=VehiclePayload(type=track.class_name, bbox=track.bbox, confidence=track.confidence),
            plate=PlatePayload(
                bbox=candidate.plate_bbox,
                detection_confidence=candidate.detection_confidence,
                crop_object_key=crop_key,
                quality_score=candidate.quality_score,
            ),
            debug=DebugPayload(frame_id=candidate.frame_id, frame_width=width, frame_height=height),
        )
