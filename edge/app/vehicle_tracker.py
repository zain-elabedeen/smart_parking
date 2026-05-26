from __future__ import annotations

import inspect
import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .schemas import Detection, TrackedVehicle

logger = logging.getLogger(__name__)


class VehicleTracker:
    """ByteTrack-backed vehicle tracker using the ``supervision`` package.

    Input:
        Created with ByteTrack lifecycle, activation, and matching settings.

    Output:
        Converts frame-level vehicle detections into stable ``TrackedVehicle``
        objects with persistent ``track_id`` values.
    """

    def __init__(
        self,
        lost_timeout_seconds: float = 2.0,
        min_track_duration_seconds: float = 0.5,
        track_activation_threshold: float = 0.25,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        frame_rate: float = 8.0,
    ) -> None:
        """Create a ByteTrack tracker.

        Args:
            lost_timeout_seconds: Approximate time after last sighting before a
                track is dropped from this wrapper's active-track cache.
            min_track_duration_seconds: Minimum age for ``mature_tracks``.
            track_activation_threshold: Detection confidence needed to start a
                track.
            minimum_matching_threshold: ByteTrack association threshold.
            minimum_consecutive_frames: Frames required before a track is
                considered active by ByteTrack.
            frame_rate: Approximate processed FPS used to size ByteTrack's lost
                track buffer.

        Returns:
            ``None``.
        """
        try:
            import supervision as sv
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("supervision is required for ByteTrack vehicle tracking") from exc

        self._sv = sv
        self._tracker = self._create_bytetrack(
            sv,
            lost_timeout_seconds=lost_timeout_seconds,
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            frame_rate=frame_rate,
        )
        self.lost_timeout_seconds = lost_timeout_seconds
        self.min_track_duration_seconds = min_track_duration_seconds
        self._tracks: dict[int, TrackedVehicle] = {}
        self._expired_tracks: list[TrackedVehicle] = []
        self._class_name_to_id: dict[str, int] = {}
        self._class_id_to_name: dict[int, str] = {}
        logger.info(
            "ByteTrack initialized lost_timeout_seconds=%.2f activation_threshold=%.2f matching_threshold=%.2f frame_rate=%.2f",
            lost_timeout_seconds,
            track_activation_threshold,
            minimum_matching_threshold,
            frame_rate,
        )

    def update(self, detections: list[Detection], timestamp: datetime) -> list[TrackedVehicle]:
        """Update ByteTrack state with detections from one frame.

        Args:
            detections: Vehicle detections for the current frame.
            timestamp: Timestamp associated with those detections.

        Returns:
            Tracks that ByteTrack reports as active for this frame.
        """
        tracked_detections = self._update_tracker(self._to_supervision_detections(detections))
        active_track_ids: set[int] = set()
        active_tracks: list[TrackedVehicle] = []

        for index in range(len(tracked_detections)):
            track_id = self._tracker_id_at(tracked_detections, index)
            if track_id is None:
                continue

            bbox = [float(value) for value in tracked_detections.xyxy[index].tolist()]
            confidence = self._confidence_at(tracked_detections, index)
            class_name = self._class_name_at(tracked_detections, index)
            previous = self._tracks.get(track_id)
            center = Detection(bbox=bbox, class_name=class_name, confidence=confidence).center

            if previous is None:
                track = TrackedVehicle(
                    track_id=track_id,
                    bbox=bbox,
                    class_name=class_name,
                    confidence=confidence,
                    first_seen_at=timestamp,
                    last_seen_at=timestamp,
                    trajectory=[center],
                )
            else:
                previous.bbox = bbox
                previous.class_name = class_name
                previous.confidence = confidence
                previous.last_seen_at = timestamp
                previous.trajectory.append(center)
                track = previous

            self._tracks[track_id] = track
            active_track_ids.add(track_id)
            active_tracks.append(track)

        self._expire_tracks(timestamp)
        return active_tracks

    def active_tracks(self) -> list[TrackedVehicle]:
        """Return tracks still retained by the wrapper.

        Returns:
            List of ``TrackedVehicle`` objects not yet expired by timeout.
        """
        return list(self._tracks.values())

    def mature_tracks(self) -> list[TrackedVehicle]:
        """Return retained tracks that have lasted long enough to be trusted.

        Returns:
            Tracks whose age is at least ``min_track_duration_seconds``.
        """
        return [
            track
            for track in self._tracks.values()
            if (track.last_seen_at - track.first_seen_at).total_seconds() >= self.min_track_duration_seconds
        ]

    def pop_expired_tracks(self) -> list[TrackedVehicle]:
        """Return and clear tracks expired during recent updates.

        Returns:
            List of tracks removed from the active cache since the previous
            call.
        """
        expired = self._expired_tracks
        self._expired_tracks = []
        return expired

    def _to_supervision_detections(self, detections: list[Detection]):
        """Convert project detection objects into ``supervision.Detections``.

        Args:
            detections: Vehicle detections from the detector.

        Returns:
            ``supervision.Detections`` with boxes, confidence, class ids, and
            class-name metadata.
        """
        if not detections:
            if hasattr(self._sv.Detections, "empty"):
                return self._sv.Detections.empty()
            return self._sv.Detections(xyxy=np.empty((0, 4)), confidence=np.array([]), class_id=np.array([], dtype=int))

        xyxy = np.array([detection.bbox for detection in detections], dtype=float)
        confidence = np.array([detection.confidence for detection in detections], dtype=float)
        class_id = np.array([self._class_id_for(detection.class_name) for detection in detections], dtype=int)
        class_names = np.array([detection.class_name for detection in detections])
        return self._sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id, data={"class_name": class_names})

    def _update_tracker(self, detections: Any):
        """Update ByteTrack across supported supervision method names.

        Args:
            detections: ``supervision.Detections`` for the current frame.

        Returns:
            Tracked ``supervision.Detections`` with tracker ids populated.
        """
        if hasattr(self._tracker, "update_with_detections"):
            return self._tracker.update_with_detections(detections)
        if hasattr(self._tracker, "update_from_detections"):
            return self._tracker.update_from_detections(detections=detections)
        if hasattr(self._tracker, "update"):
            return self._tracker.update(detections=detections)
        raise RuntimeError("installed supervision ByteTrack object has no supported update method")

    def _class_id_for(self, class_name: str) -> int:
        """Return a stable integer class id for a class name.

        Args:
            class_name: Detector class label.

        Returns:
            Integer class id used inside ``supervision.Detections``.
        """
        existing = self._class_name_to_id.get(class_name)
        if existing is not None:
            return existing

        class_id = len(self._class_name_to_id)
        self._class_name_to_id[class_name] = class_id
        self._class_id_to_name[class_id] = class_name
        return class_id

    def _tracker_id_at(self, detections: Any, index: int) -> int | None:
        """Read one tracker id from a ``supervision.Detections`` object.

        Args:
            detections: Tracked ``supervision.Detections`` result.
            index: Detection index to read.

        Returns:
            Integer tracker id, or ``None`` if ByteTrack did not assign one.
        """
        tracker_ids = getattr(detections, "tracker_id", None)
        if tracker_ids is None:
            return None
        raw = tracker_ids[index]
        return None if raw is None else int(raw)

    def _confidence_at(self, detections: Any, index: int) -> float:
        """Read one confidence value from tracked detections.

        Args:
            detections: Tracked ``supervision.Detections`` result.
            index: Detection index to read.

        Returns:
            Detection confidence as a float, or 0.0 when unavailable.
        """
        confidence = getattr(detections, "confidence", None)
        if confidence is None:
            return 0.0
        return float(confidence[index])

    def _class_name_at(self, detections: Any, index: int) -> str:
        """Read one class name from tracked detections.

        Args:
            detections: Tracked ``supervision.Detections`` result.
            index: Detection index to read.

        Returns:
            Original class name when metadata is present; otherwise a fallback
            name derived from class id.
        """
        data = getattr(detections, "data", {}) or {}
        class_names = data.get("class_name")
        if class_names is not None and len(class_names) > index:
            return str(class_names[index])

        class_ids = getattr(detections, "class_id", None)
        if class_ids is None:
            return "vehicle"
        return self._class_id_to_name.get(int(class_ids[index]), "vehicle")

    def _expire_tracks(self, timestamp: datetime) -> None:
        """Remove wrapper-cached tracks that have not been seen recently.

        Args:
            timestamp: Current frame timestamp.

        Returns:
            ``None``.
        """
        lost_after = timedelta(seconds=self.lost_timeout_seconds)
        expired = [track_id for track_id, track in self._tracks.items() if timestamp - track.last_seen_at > lost_after]
        for track_id in expired:
            logger.info("track expired track_id=%s", track_id)
            self._expired_tracks.append(self._tracks.pop(track_id))

    def _create_bytetrack(
        self,
        sv: Any,
        *,
        lost_timeout_seconds: float,
        track_activation_threshold: float,
        minimum_matching_threshold: float,
        minimum_consecutive_frames: int,
        frame_rate: float,
    ):
        """Instantiate ``supervision.ByteTrack`` across supported APIs.

        Args:
            sv: Imported ``supervision`` module.
            lost_timeout_seconds: Desired lost-track timeout in seconds.
            track_activation_threshold: Detection score needed to start tracks.
            minimum_matching_threshold: Association threshold.
            minimum_consecutive_frames: Consecutive-frame activation setting.
            frame_rate: Approximate processed frames per second.

        Returns:
            Configured ``supervision.ByteTrack`` instance.
        """
        if not hasattr(sv, "ByteTrack"):
            raise RuntimeError("installed supervision package does not expose ByteTrack")

        parameters = inspect.signature(sv.ByteTrack).parameters
        kwargs: dict[str, Any] = {}
        lost_track_buffer = max(1, int(round(lost_timeout_seconds * 30.0)))

        if "track_activation_threshold" in parameters:
            kwargs["track_activation_threshold"] = track_activation_threshold
        elif "track_thresh" in parameters:
            kwargs["track_thresh"] = track_activation_threshold

        if "lost_track_buffer" in parameters:
            kwargs["lost_track_buffer"] = lost_track_buffer
        elif "track_buffer" in parameters:
            kwargs["track_buffer"] = lost_track_buffer

        if "minimum_matching_threshold" in parameters:
            kwargs["minimum_matching_threshold"] = minimum_matching_threshold
        elif "match_thresh" in parameters:
            kwargs["match_thresh"] = minimum_matching_threshold

        if "minimum_consecutive_frames" in parameters:
            kwargs["minimum_consecutive_frames"] = minimum_consecutive_frames

        if "frame_rate" in parameters:
            kwargs["frame_rate"] = int(round(frame_rate))

        return sv.ByteTrack(**kwargs)
