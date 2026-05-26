from __future__ import annotations

import numpy as np

from app import video_reader
from app.video_reader import VideoReader, _is_reconnectable_source, _normalize_source


def test_normalize_source_converts_webcam_uri() -> None:
    """Verify webcam URI sources become OpenCV integer indexes.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    assert _normalize_source("webcam://0") == 0


def test_file_sources_are_not_reconnectable() -> None:
    """Verify local file paths stop at EOF instead of reconnecting.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    assert _is_reconnectable_source("demo_videos/entry.mp4") is False
    assert _is_reconnectable_source("/tmp/entry.mp4") is False


def test_live_sources_are_reconnectable() -> None:
    """Verify webcams and RTSP streams reconnect after read failures.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    assert _is_reconnectable_source(0) is True
    assert _is_reconnectable_source("rtsp://camera/live") is True


def test_file_reader_stops_at_eof(monkeypatch, tmp_path) -> None:
    """Verify local file iteration exits when OpenCV reports EOF.

    Args:
        monkeypatch: Pytest fixture used to replace OpenCV capture.
        tmp_path: Pytest fixture providing a temporary file path.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    video_path = tmp_path / "entry.mp4"
    video_path.write_bytes(b"placeholder")
    captures: list[FakeCapture] = []

    class FakeCapture:
        """Small stand-in for ``cv2.VideoCapture`` used by this test.

        Input:
            Created with a video source.

        Output:
            Returns two image frames, then EOF.
        """

        def __init__(self, source: str) -> None:
            """Create a fake capture with two readable frames.

            Args:
                source: Video source path passed by ``VideoReader``.

            Returns:
                ``None``.
            """
            self.source = source
            self.reads = 0
            self.released = False
            captures.append(self)

        def isOpened(self) -> bool:
            """Return whether the fake capture opened successfully.

            Returns:
                Always ``True`` for this test.
            """
            return True

        def read(self):
            """Return two fake frames before reporting EOF.

            Returns:
                ``(True, image)`` for the first two reads, then
                ``(False, None)``.
            """
            self.reads += 1
            if self.reads <= 2:
                return True, np.zeros((4, 4, 3), dtype=np.uint8)
            return False, None

        def get(self, prop_id: int) -> float:
            """Return basic source metadata for log messages.

            Args:
                prop_id: OpenCV capture property id.

            Returns:
                FPS, frame count, or ``0.0``.
            """
            if prop_id == video_reader.cv2.CAP_PROP_FPS:
                return 30.0
            if prop_id == video_reader.cv2.CAP_PROP_FRAME_COUNT:
                return 2.0
            return 0.0

        def release(self) -> None:
            """Mark the fake capture as released.

            Returns:
                ``None``.
            """
            self.released = True

    monkeypatch.setattr(video_reader.cv2, "VideoCapture", FakeCapture)

    frames = list(VideoReader(str(video_path), site_id="site", camera_id="cam").frames())

    assert len(frames) == 2
    assert captures[0].released is True
