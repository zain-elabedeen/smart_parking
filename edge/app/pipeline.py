from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .config import AppConfig
from .crop_selector import PlateCropCandidate, PlateCropSelector
from .direction_estimator import DirectionEvent, LineCrossingDirectionEstimator
from .event_service import PlateCropEventPublisher
from .frame_sampler import FrameSampler
from .image_utils import crop_image, offset_bbox, round_bbox
from .plate_detector import PlateDetector
from .processing import FrameProcessingStats, PlateDetectionPolicy, summarize_classes
from .schemas import Direction, Frame, TrackedVehicle
from .vehicle_detector import VehicleDetector
from .vehicle_tracker import VehicleTracker
from .video_reader import VideoReader

logger = logging.getLogger(__name__)
HIGH_QUALITY_EVENT_THRESHOLD = 0.85


@dataclass(slots=True)
class EdgePipeline:
    """Coordinates the frame-by-frame parking edge-node workflow.

    Input fields:
        config: Validated runtime configuration.
        reader: Video frame source.
        sampler: Frame sampling policy.
        vehicle_detector: Vehicle detector implementation.
        tracker: Vehicle tracker implementation.
        plate_detector: License plate detector implementation.
        direction: Direction estimator implementation.
        selector: Per-track best-crop selector.
        event_publisher: Event persistence and queueing service.
        metrics: Metrics registry object.
        verbose_tracks: Whether to emit detailed per-track logs.

    Output:
        Processes frames, saves selected crop images, and publishes queued
        ``PlateCropDetected`` events.
    """

    config: AppConfig
    reader: VideoReader
    sampler: FrameSampler
    vehicle_detector: VehicleDetector
    tracker: VehicleTracker
    plate_detector: PlateDetector
    direction: LineCrossingDirectionEstimator
    selector: PlateCropSelector
    event_publisher: PlateCropEventPublisher
    metrics: Any
    verbose_tracks: bool = False
    published_track_ids: set[int] = field(default_factory=set)
    plate_policy: PlateDetectionPolicy = field(init=False)

    def __post_init__(self) -> None:
        """Initialize derived pipeline collaborators.

        Returns:
            ``None``.
        """
        self.plate_policy = PlateDetectionPolicy(self.config.processing)

    def run(self, *, max_frames: int = 0) -> int:
        """Run the processing loop until EOF, stream stop, or frame limit.

        Args:
            max_frames: Number of sampled frames to process before stopping;
                ``0`` means no explicit limit.

        Returns:
            Number of sampled frames processed.
        """
        processed = 0
        for frame in self.reader.frames():
            self._record_frame_read()
            if not self.sampler.should_process(frame.timestamp):
                continue

            processed += 1
            self._record_frame_processed()
            self._process_sampled_frame(frame, processed)

            if max_frames and processed >= max_frames:
                logger.info("max_frames reached processed=%s; exiting", processed)
                break

        logger.info("processing loop finished processed_frames=%s", processed)
        return processed

    def _record_frame_read(self) -> None:
        """Update metrics after reading a frame.

        Returns:
            ``None``.
        """
        if self.metrics.enabled:
            self.metrics.frames_read.inc()
            self.metrics.camera_connected.set(1 if self.reader.camera_connected else 0)

    def _record_frame_processed(self) -> None:
        """Update metrics after selecting a frame for inference.

        Returns:
            ``None``.
        """
        if self.metrics.enabled:
            self.metrics.frames_processed.inc()

    def _process_sampled_frame(self, frame: Frame, processed_index: int) -> None:
        """Run detection, tracking, plate selection, and expiry handling.

        Args:
            frame: Sampled frame selected for inference.
            processed_index: One-based sampled-frame counter.

        Returns:
            ``None``.
        """
        vehicles = self.vehicle_detector.detect(frame)
        should_log_frame = processed_index % self.config.processing.frame_log_interval == 0
        self._log_vehicle_summary(frame, vehicles, should_log_frame)

        tracks = self.tracker.update(vehicles, frame.timestamp)
        if should_log_frame:
            logger.info("frame=%s active_tracks=%s track_ids=%s", frame.frame_id, len(tracks), [track.track_id for track in tracks])

        stats = FrameProcessingStats()
        for track in tracks:
            self._process_track(frame, track, stats)

        if should_log_frame:
            logger.info(
                "frame=%s plate_attempts=%s plate_detections=%s skipped_static=%s skipped_class=%s",
                frame.frame_id,
                stats.plate_attempts,
                stats.plate_detections,
                stats.skipped_static,
                stats.skipped_class,
            )

        self._handle_expired_tracks(frame)

    def _log_vehicle_summary(self, frame: Frame, vehicles: list, should_log_frame: bool) -> None:
        """Log compact vehicle detection counts for one frame.

        Args:
            frame: Current sampled frame.
            vehicles: Vehicle detections returned by the detector.
            should_log_frame: Whether this frame is due for summary logging.

        Returns:
            ``None``.
        """
        if not should_log_frame:
            return
        if vehicles:
            logger.info(
                "frame=%s vehicles_detected=%s classes=%s",
                frame.frame_id,
                len(vehicles),
                summarize_classes(vehicle.class_name for vehicle in vehicles),
            )
        else:
            logger.info("frame=%s vehicles_detected=0", frame.frame_id)

    def _process_track(self, frame: Frame, track: TrackedVehicle, stats: FrameProcessingStats) -> None:
        """Process one active track for direction, plate crops, and events.

        Args:
            frame: Current sampled frame.
            track: Current tracked vehicle state.
            stats: Per-frame counters to update.

        Returns:
            ``None``.
        """
        direction_event = self.direction.update(track)
        if direction_event is not None:
            logger.info(
                "track=%s crossed line direction=%s crossed_at=%s",
                direction_event.track_id,
                direction_event.direction,
                direction_event.crossed_at.isoformat(),
            )

        self._maybe_detect_plate(frame, track, stats)
        self._maybe_publish_active_track(frame, track, direction_event)

    def _maybe_detect_plate(self, frame: Frame, track: TrackedVehicle, stats: FrameProcessingStats) -> None:
        """Run plate detection for a track when policy allows it.

        Args:
            frame: Current sampled frame.
            track: Current tracked vehicle state.
            stats: Per-frame counters to update.

        Returns:
            ``None``.
        """
        decision = self.plate_policy.evaluate(track)
        if not decision.should_detect:
            stats.record_skip(decision.skip_reason)
            if self.verbose_tracks:
                logger.info(
                    "track=%s skipped_plate_detection reason=%s class=%s displacement_px=%.1f",
                    track.track_id,
                    decision.skip_reason,
                    track.class_name,
                    self.plate_policy.track_displacement(track),
                )
            return

        vehicle_crop = crop_image(frame.image, track.bbox)
        if self.verbose_tracks:
            logger.info("track=%s vehicle_crop shape=%s bbox=%s", track.track_id, getattr(vehicle_crop, "shape", None), round_bbox(track.bbox))

        plate_detections = self.plate_detector.detect(vehicle_crop)
        stats.record_plate_attempt(len(plate_detections))
        if self.verbose_tracks:
            logger.info("track=%s plate_detections=%s", track.track_id, len(plate_detections))

        for plate in plate_detections:
            previous_best = self.selector.best_for_track(track.track_id)
            candidate = self.selector.add_candidate(
                PlateCropCandidate(
                    track_id=track.track_id,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    plate_bbox=offset_bbox(plate.bbox, track.bbox[0], track.bbox[1]),
                    vehicle_bbox=track.bbox,
                    crop=plate.crop,
                    detection_confidence=plate.confidence,
                )
            )
            candidate_is_new_best = previous_best is None or candidate.quality_score >= previous_best.quality_score
            if self.verbose_tracks or candidate_is_new_best:
                logger.info(
                    "track=%s plate_candidate confidence=%.3f quality=%.3f new_best=%s plate_bbox=%s crop_shape=%s",
                    track.track_id,
                    candidate.detection_confidence,
                    candidate.quality_score,
                    candidate_is_new_best,
                    round_bbox(candidate.plate_bbox),
                    getattr(candidate.crop, "shape", None),
                )

    def _maybe_publish_active_track(self, frame: Frame, track: TrackedVehicle, direction_event: DirectionEvent | None) -> None:
        """Publish an active track when it has a crop and known direction.

        Args:
            frame: Current sampled frame.
            track: Current tracked vehicle state.
            direction_event: Optional crossing event emitted for this track.

        Returns:
            ``None``.
        """
        best = self.selector.best_for_track(track.track_id)
        if track.track_id in self.published_track_ids or best is None:
            return

        publish_decision = self._active_track_publish_decision(track, best, direction_event, frame.timestamp)
        if publish_decision is None:
            return

        direction_value, event_timestamp = publish_decision
        if self.event_publisher.publish(
            frame=frame,
            track=track,
            direction_value=direction_value,
            event_timestamp=event_timestamp,
            candidate=best,
        ):
            self.published_track_ids.add(track.track_id)
            self.selector.forget(track.track_id)

    def _active_track_publish_decision(
        self,
        track: TrackedVehicle,
        best: PlateCropCandidate,
        direction_event: DirectionEvent | None,
        fallback_timestamp: datetime,
    ) -> tuple[Direction, datetime] | None:
        """Decide whether an active track is ready to publish.

        Args:
            track: Current tracked vehicle state.
            best: Current best plate crop candidate for the track.
            direction_event: Optional crossing event emitted for this track.
            fallback_timestamp: Current frame timestamp.

        Returns:
            ``(direction, timestamp)`` when the track should publish, otherwise
            ``None``.
        """
        if direction_event is not None:
            return direction_event.direction, direction_event.crossed_at

        known_direction = self.direction.direction_for_track(track.track_id)
        if best.quality_score >= HIGH_QUALITY_EVENT_THRESHOLD and known_direction != "UNKNOWN":
            return known_direction, best.timestamp

        _ = fallback_timestamp
        return None

    def _handle_expired_tracks(self, frame: Frame) -> None:
        """Publish or clean up tracks that ByteTrack has expired.

        Args:
            frame: Current sampled frame, used for event debug dimensions.

        Returns:
            ``None``.
        """
        for expired_track in self.tracker.pop_expired_tracks():
            best = self.selector.pop_best(expired_track.track_id)
            try:
                logger.info(
                    "track=%s expired age_seconds=%.2f has_best_crop=%s already_published=%s",
                    expired_track.track_id,
                    (expired_track.last_seen_at - expired_track.first_seen_at).total_seconds(),
                    best is not None,
                    expired_track.track_id in self.published_track_ids,
                )
                if self._should_publish_expired_track(expired_track, best):
                    self._publish_expired_track(frame, expired_track, best)
            finally:
                self.direction.forget(expired_track.track_id)
                self.selector.forget(expired_track.track_id)

    def _should_publish_expired_track(self, track: TrackedVehicle, best: PlateCropCandidate | None) -> bool:
        """Return whether an expired track should produce an event.

        Args:
            track: Expired tracked vehicle.
            best: Best crop candidate retained for the track, if any.

        Returns:
            ``True`` when the track has not already published, has a crop, and
            was visible for the configured minimum duration.
        """
        age_seconds = (track.last_seen_at - track.first_seen_at).total_seconds()
        return (
            track.track_id not in self.published_track_ids
            and best is not None
            and age_seconds >= self.config.tracker.min_track_duration_seconds
        )

    def _publish_expired_track(self, frame: Frame, track: TrackedVehicle, best: PlateCropCandidate | None) -> None:
        """Publish the best crop for an expired track.

        Args:
            frame: Current sampled frame, used for event debug dimensions.
            track: Expired tracked vehicle.
            best: Best crop candidate retained for the track.

        Returns:
            ``None``.
        """
        if best is None:
            return
        if self.event_publisher.publish(
            frame=frame,
            track=track,
            direction_value=self.direction.direction_for_track(track.track_id),
            event_timestamp=track.last_seen_at,
            candidate=best,
        ):
            self.published_track_ids.add(track.track_id)
