from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

import numpy as np

from app.schemas import Detection
from app.vehicle_tracker import VehicleTracker


class FakeDetections:
    """Minimal ``supervision.Detections`` replacement for tracker tests.

    Input:
        Created with arrays matching the subset used by ``VehicleTracker``.

    Output:
        Behaves like a tracked detections container in unit tests.
    """

    def __init__(self, xyxy, confidence=None, class_id=None, data=None, tracker_id=None) -> None:
        """Create a fake detections object.

        Args:
            xyxy: Bounding boxes array.
            confidence: Optional confidence array.
            class_id: Optional class id array.
            data: Optional metadata dictionary.
            tracker_id: Optional tracker id array.

        Returns:
            ``None``.
        """
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.data = data or {}
        self.tracker_id = tracker_id

    def __len__(self) -> int:
        """Return the number of detections.

        Returns:
            Detection count.
        """
        return len(self.xyxy)

    @classmethod
    def empty(cls):
        """Create an empty fake detections object.

        Returns:
            Empty ``FakeDetections`` instance.
        """
        return cls(xyxy=np.empty((0, 4)), confidence=np.array([]), class_id=np.array([], dtype=int), tracker_id=np.array([]))


class FakeByteTrack:
    """Minimal ByteTrack replacement that assigns deterministic ids.

    Input:
        Accepts the same keyword names used by recent supervision versions.

    Output:
        Returns detections with a populated ``tracker_id`` array.
    """

    def __init__(
        self,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        minimum_consecutive_frames=1,
        frame_rate=8,
    ) -> None:
        """Capture constructor settings for compatibility testing.

        Args:
            track_activation_threshold: Activation threshold.
            lost_track_buffer: Lost track buffer.
            minimum_matching_threshold: Matching threshold.
            minimum_consecutive_frames: Consecutive frame setting.
            frame_rate: Frame-rate setting.

        Returns:
            ``None``.
        """
        self.settings = {
            "track_activation_threshold": track_activation_threshold,
            "lost_track_buffer": lost_track_buffer,
            "minimum_matching_threshold": minimum_matching_threshold,
            "minimum_consecutive_frames": minimum_consecutive_frames,
            "frame_rate": frame_rate,
        }

    def update_with_detections(self, detections: FakeDetections) -> FakeDetections:
        """Assign deterministic tracker ids to detections.

        Args:
            detections: Fake detections from the wrapper conversion.

        Returns:
            Same detections object with ``tracker_id`` populated.
        """
        detections.tracker_id = np.arange(1, len(detections) + 1)
        return detections


def test_vehicle_tracker_uses_supervision_bytetrack(monkeypatch) -> None:
    """Verify the wrapper converts detections and returns ByteTrack ids.

    Args:
        monkeypatch: Pytest fixture used to install a fake supervision module.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    fake_supervision = types.SimpleNamespace(ByteTrack=FakeByteTrack, Detections=FakeDetections)
    monkeypatch.setitem(sys.modules, "supervision", fake_supervision)
    tracker = VehicleTracker(frame_rate=8)

    tracks = tracker.update(
        [Detection(bbox=[10, 20, 100, 200], class_name="car", confidence=0.9)],
        datetime.now(tz=timezone.utc),
    )

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].class_name == "car"
    assert tracks[0].bbox == [10.0, 20.0, 100.0, 200.0]

