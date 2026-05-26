from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected in runtime/test envs
    np = Any  # type: ignore

from pydantic import BaseModel, Field


Direction = Literal["ENTRY", "EXIT", "UNKNOWN"]


def utc_now() -> datetime:
    """Return the current UTC time.

    Returns:
        A timezone-aware ``datetime`` set to UTC.
    """
    return datetime.now(tz=timezone.utc)


def stable_event_id(*parts: object) -> str:
    """Create a deterministic event id from immutable event facts.

    Args:
        *parts: Values that uniquely identify the event, such as site, camera,
            track id, and crossing timestamp.

    Returns:
        A stable 32-character hexadecimal id that can be reused safely on retry.
    """
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class Frame:
    """A single video frame plus camera/site metadata.

    Input fields:
        frame_id: Unique id for this frame within the reader session.
        camera_id: Camera identifier from configuration.
        site_id: Parking site identifier from configuration.
        timestamp: Capture/read timestamp.
        image: OpenCV image array in BGR channel order.

    Output:
        Passed between reader, sampler, detector, and event-building code.
    """

    frame_id: str
    camera_id: str
    site_id: str
    timestamp: datetime
    image: "np.ndarray"


@dataclass(slots=True)
class Point:
    """A 2D coordinate used for centroids and trajectories.

    Input fields:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.

    Output:
        Used by tracking and line-crossing direction estimation.
    """

    x: float
    y: float


@dataclass(slots=True)
class Detection:
    """A raw object-detection result before track assignment.

    Input fields:
        bbox: Bounding box as ``[x1, y1, x2, y2]`` in frame coordinates.
        class_name: Detector class label, such as ``car`` or ``truck``.
        confidence: Detector confidence from 0.0 to 1.0.

    Output:
        Consumed by the tracker and converted into ``TrackedVehicle`` objects.
    """

    bbox: list[float]
    class_name: str
    confidence: float

    @property
    def width(self) -> float:
        """Return the bounding-box width in pixels.

        Returns:
            Non-negative width computed from ``x2 - x1``.
        """
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        """Return the bounding-box height in pixels.

        Returns:
            Non-negative height computed from ``y2 - y1``.
        """
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> Point:
        """Return the center point of the detection box.

        Returns:
            ``Point`` at the midpoint of the bounding box.
        """
        return Point((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


@dataclass(slots=True)
class TrackedVehicle:
    """A vehicle detection with a stable track id across frames.

    Input fields:
        track_id: Stable id assigned by the tracker.
        bbox: Current vehicle box as ``[x1, y1, x2, y2]``.
        class_name: Vehicle class label.
        confidence: Latest detector confidence.
        first_seen_at: Timestamp when the track was created.
        last_seen_at: Timestamp when the track was last updated.
        trajectory: List of centroid points observed for this track.

    Output:
        Used by plate detection, direction estimation, and event construction.
    """

    track_id: int
    bbox: list[float]
    class_name: str
    confidence: float
    first_seen_at: datetime
    last_seen_at: datetime
    trajectory: list[Point] = field(default_factory=list)

    @property
    def center(self) -> Point:
        """Return the current vehicle center point.

        Returns:
            ``Point`` at the midpoint of the current vehicle bounding box.
        """
        return Point((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


@dataclass(slots=True)
class PlateDetection:
    """A license-plate detection inside a vehicle crop.

    Input fields:
        bbox: Plate box as ``[x1, y1, x2, y2]`` relative to the vehicle crop.
        confidence: Plate detector confidence from 0.0 to 1.0.
        crop: Cropped plate image array.

    Output:
        Converted into scored crop candidates and later event payloads.
    """

    bbox: list[float]
    confidence: float
    crop: "np.ndarray"


class VehiclePayload(BaseModel):
    """Vehicle section of a ``PlateCropDetected`` cloud event.

    Input fields:
        type: Vehicle class label.
        bbox: Vehicle box in full-frame coordinates.
        confidence: Detector confidence from 0.0 to 1.0.

    Output:
        Serialized into the outgoing event JSON.
    """

    type: str
    bbox: list[float]
    confidence: float = Field(ge=0.0, le=1.0)


class PlatePayload(BaseModel):
    """License-plate section of a ``PlateCropDetected`` cloud event.

    Input fields:
        bbox: Plate box in full-frame coordinates.
        detection_confidence: Plate detector confidence from 0.0 to 1.0.
        crop_object_key: Object-storage key for the saved crop image.
        quality_score: Selected crop quality score from 0.0 to 1.0.

    Output:
        Serialized into the outgoing event JSON.
    """

    bbox: list[float]
    detection_confidence: float = Field(ge=0.0, le=1.0)
    crop_object_key: str
    quality_score: float = Field(ge=0.0, le=1.0)


class DebugPayload(BaseModel):
    """Debug metadata attached to a ``PlateCropDetected`` event.

    Input fields:
        frame_id: Source frame id for the selected crop.
        frame_width: Width of the full frame in pixels.
        frame_height: Height of the full frame in pixels.

    Output:
        Serialized into the outgoing event JSON for traceability.
    """

    frame_id: str
    frame_width: int
    frame_height: int


class PlateCropDetected(BaseModel):
    """Cloud event emitted when the edge node selects a plate crop.

    Input fields:
        event_id: Stable idempotency key.
        event_type: Event name; always ``PlateCropDetected``.
        schema_version: Event schema version.
        site_id: Parking site identifier.
        camera_id: Camera identifier.
        edge_device_id: Edge device identifier.
        track_id: Vehicle tracker id.
        timestamp: Event timestamp.
        direction: Vehicle direction, ``ENTRY``, ``EXIT``, or ``UNKNOWN``.
        vehicle: Vehicle detection payload.
        plate: Plate crop payload.
        debug: Debug metadata payload.

    Output:
        Serialized JSON message for Kafka or another cloud gateway.
    """

    event_id: str
    event_type: Literal["PlateCropDetected"] = "PlateCropDetected"
    schema_version: str = "1.0"
    site_id: str
    camera_id: str
    edge_device_id: str
    track_id: int
    timestamp: datetime
    direction: Direction
    vehicle: VehiclePayload
    plate: PlatePayload
    debug: DebugPayload

    def to_json(self) -> str:
        """Serialize the event to a JSON string.

        Returns:
            JSON string compatible with Pydantic v2 and v1 runtimes.
        """
        if hasattr(self, "model_dump_json"):
            return self.model_dump_json()
        return self.json()  # pragma: no cover - pydantic v1 compatibility

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event to a plain dictionary.

        Returns:
            JSON-compatible dictionary compatible with Pydantic v2 and v1.
        """
        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return json.loads(self.json())  # pragma: no cover - pydantic v1 compatibility
