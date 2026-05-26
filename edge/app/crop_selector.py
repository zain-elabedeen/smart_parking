from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - exercised only without OpenCV installed
    cv2 = None  # type: ignore


@dataclass(slots=True)
class PlateCropCandidate:
    """A scored candidate plate crop observed for a vehicle track.

    Input fields:
        track_id: Vehicle track that produced the crop.
        frame_id: Source frame id for traceability.
        timestamp: Timestamp when the crop was observed.
        plate_bbox: Plate box in full-frame coordinates.
        vehicle_bbox: Vehicle box in full-frame coordinates.
        crop: Plate crop image.
        detection_confidence: Plate detector confidence.
        quality_score: Computed crop quality score; set by the selector.
        object_key: Optional storage key after the crop is persisted.
        angle_score: Optional normalized angle score.

    Output:
        Stored in a per-track buffer and eventually serialized into an event.
    """

    track_id: int
    frame_id: str
    timestamp: datetime
    plate_bbox: list[float]
    vehicle_bbox: list[float]
    crop: "np.ndarray"
    detection_confidence: float
    quality_score: float = 0.0
    object_key: str | None = None
    angle_score: float = 1.0


class TrackPlateBuffer:
    """Per-track buffer that keeps the best plate crop candidates.

    Input:
        Created with a track id and maximum number of candidates to retain.

    Output:
        Maintains sorted candidate state for one vehicle track.
    """

    def __init__(self, track_id: int, max_candidates: int = 20) -> None:
        """Create an empty candidate buffer for one track.

        Args:
            track_id: Vehicle track id this buffer belongs to.
            max_candidates: Maximum candidates retained after sorting.

        Returns:
            ``None``.
        """
        self.track_id = track_id
        self.max_candidates = max_candidates
        self.candidates: list[PlateCropCandidate] = []
        self.best_candidate: PlateCropCandidate | None = None

    def add_candidate(self, candidate: PlateCropCandidate) -> PlateCropCandidate:
        """Score and store one candidate crop.

        Args:
            candidate: Candidate crop to score and add.

        Returns:
            The same candidate instance with ``quality_score`` populated.
        """
        candidate.quality_score = score_candidate(candidate)
        self.candidates.append(candidate)
        self.candidates.sort(key=lambda item: item.quality_score, reverse=True)
        del self.candidates[self.max_candidates :]
        self.best_candidate = self.candidates[0]
        return candidate


class PlateCropSelector:
    """Manages plate crop buffers for all active vehicle tracks.

    Input:
        Created with a maximum candidate count per track.

    Output:
        Provides best-crop lookup and removal for tracks that publish events.
    """

    def __init__(self, max_candidates_per_track: int = 20) -> None:
        """Create a selector with no active track buffers.

        Args:
            max_candidates_per_track: Maximum candidates retained per track.

        Returns:
            ``None``.
        """
        self._buffers: dict[int, TrackPlateBuffer] = {}
        self.max_candidates_per_track = max_candidates_per_track

    def add_candidate(self, candidate: PlateCropCandidate) -> PlateCropCandidate:
        """Add a plate crop candidate to the matching track buffer.

        Args:
            candidate: Candidate crop produced by plate detection.

        Returns:
            The same candidate instance with an updated quality score.
        """
        buffer = self._buffers.setdefault(
            candidate.track_id,
            TrackPlateBuffer(candidate.track_id, max_candidates=self.max_candidates_per_track),
        )
        return buffer.add_candidate(candidate)

    def best_for_track(self, track_id: int) -> PlateCropCandidate | None:
        """Return the current best candidate for one track without removing it.

        Args:
            track_id: Vehicle track id to inspect.

        Returns:
            Best candidate if the track has candidates, otherwise ``None``.
        """
        buffer = self._buffers.get(track_id)
        return None if buffer is None else buffer.best_candidate

    def pop_best(self, track_id: int) -> PlateCropCandidate | None:
        """Return and remove the best candidate for one track.

        Args:
            track_id: Vehicle track id to remove from the selector.

        Returns:
            Best candidate if present, otherwise ``None``.
        """
        buffer = self._buffers.pop(track_id, None)
        return None if buffer is None else buffer.best_candidate

    def forget(self, track_id: int) -> None:
        """Drop all crop candidates for one track.

        Args:
            track_id: Vehicle track id to remove.

        Returns:
            ``None``.
        """
        self._buffers.pop(track_id, None)


def score_candidate(candidate: PlateCropCandidate) -> float:
    """Compute a normalized quality score for a plate crop.

    Args:
        candidate: Plate crop candidate containing bbox, confidence, crop, and
            optional angle information.

    Returns:
        Score from 0.0 to 1.0 where higher means a better crop for OCR.
    """
    plate_width = max(0.0, candidate.plate_bbox[2] - candidate.plate_bbox[0])
    plate_height = max(0.0, candidate.plate_bbox[3] - candidate.plate_bbox[1])
    normalized_plate_size = min(1.0, sqrt(max(0.0, plate_width * plate_height) / (160.0 * 48.0)))
    sharpness = normalized_sharpness(candidate.crop)
    center = center_score(candidate.plate_bbox, candidate.vehicle_bbox)
    angle = min(1.0, max(0.0, candidate.angle_score))

    score = (
        0.40 * _clamp01(candidate.detection_confidence)
        + 0.25 * normalized_plate_size
        + 0.20 * sharpness
        + 0.10 * center
        + 0.05 * angle
    )
    return round(_clamp01(score), 6)


def normalized_sharpness(crop: "np.ndarray") -> float:
    """Convert Laplacian sharpness into a normalized score.

    Args:
        crop: Plate crop image array.

    Returns:
        Sharpness score from 0.0 to 1.0.
    """
    variance = laplacian_variance(crop)
    return _clamp01(variance / 500.0)


def laplacian_variance(crop: "np.ndarray") -> float:
    """Measure image sharpness using variance of the Laplacian.

    Args:
        crop: Plate crop image array, grayscale or color.

    Returns:
        Non-negative variance value; higher generally means sharper edges.
    """
    if crop.size == 0:
        return 0.0

    if cv2 is not None:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    gray = crop.astype(float)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    laplacian = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(laplacian.var())


def center_score(plate_bbox: list[float], vehicle_bbox: list[float]) -> float:
    """Score how close a plate is to the center of its vehicle box.

    Args:
        plate_bbox: Plate box in full-frame coordinates.
        vehicle_bbox: Vehicle box in full-frame coordinates.

    Returns:
        Score from 0.0 to 1.0 where 1.0 means perfectly centered.
    """
    px = (plate_bbox[0] + plate_bbox[2]) / 2.0
    py = (plate_bbox[1] + plate_bbox[3]) / 2.0
    vx = (vehicle_bbox[0] + vehicle_bbox[2]) / 2.0
    vy = (vehicle_bbox[1] + vehicle_bbox[3]) / 2.0
    vw = max(1.0, vehicle_bbox[2] - vehicle_bbox[0])
    vh = max(1.0, vehicle_bbox[3] - vehicle_bbox[1])
    normalized_distance = sqrt(((px - vx) / vw) ** 2 + ((py - vy) / vh) ** 2) * 2.0
    return _clamp01(1.0 - normalized_distance)


def _clamp01(value: Any) -> float:
    """Clamp a numeric value to the inclusive range 0.0 through 1.0.

    Args:
        value: Number-like value to clamp.

    Returns:
        Clamped float.
    """
    return min(1.0, max(0.0, float(value)))
