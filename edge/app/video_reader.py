from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency
    cv2 = None  # type: ignore

from .schemas import Frame

logger = logging.getLogger(__name__)


class VideoReader:
    """OpenCV-backed reader for RTSP streams, local files, or webcams.

    Input:
        Created with a video source and camera/site identifiers.

    Output:
        Yields ``Frame`` objects from the configured source.
    """

    def __init__(
        self,
        source: str,
        *,
        site_id: str,
        camera_id: str,
        reconnect_delay_seconds: float = 5.0,
        max_read_failures: int = 3,
    ) -> None:
        """Create a video reader with reconnect behavior.

        Args:
            source: RTSP URL, local file path, ``file://`` URI, or
                ``webcam://N`` URI.
            site_id: Parking site identifier added to emitted frames.
            camera_id: Camera identifier added to emitted frames.
            reconnect_delay_seconds: Delay before reconnect attempts.
            max_read_failures: Consecutive failed reads before reconnecting.

        Returns:
            ``None``.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV is required for video ingestion")
        self.source = _normalize_source(source)
        self.reconnect_on_read_failure = _is_reconnectable_source(self.source)
        if not self.reconnect_on_read_failure and not Path(str(self.source)).exists():
            raise FileNotFoundError(f"video file not found: {self.source}")
        self.site_id = site_id
        self.camera_id = camera_id
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self.max_read_failures = max_read_failures
        self.camera_connected = False
        self._cap = None
        self._frame_number = 0

    def frames(self) -> Iterator[Frame]:
        """Yield frames, reconnecting only for live sources.

        Returns:
            Iterator of ``Frame`` objects. For files, iteration stops at EOF.
            For RTSP/webcam sources, repeated read failures trigger reconnect.
        """
        failures = 0
        while True:
            if self._cap is None:
                self._connect()

            ok, image = self._cap.read()
            if not ok:
                if not self.reconnect_on_read_failure:
                    logger.info("video file reached end source=%s frames_read=%s", self.source, self._frame_number)
                    self._release()
                    break

                failures += 1
                self.camera_connected = False
                if failures >= self.max_read_failures:
                    logger.warning("video read failed source=%s failures=%s; reconnecting", self.source, failures)
                    self._release()
                    time.sleep(self.reconnect_delay_seconds)
                    failures = 0
                continue

            failures = 0
            self.camera_connected = True
            self._frame_number += 1
            yield Frame(
                frame_id=f"frame-{self._frame_number:06d}",
                site_id=self.site_id,
                camera_id=self.camera_id,
                timestamp=datetime.now(tz=timezone.utc),
                image=image,
            )

    def _connect(self) -> None:
        """Open the configured video source.

        Returns:
            ``None``. Updates ``camera_connected`` and internal capture state.
        """
        logger.info("connecting video source=%s", self.source)
        self._cap = cv2.VideoCapture(self.source)
        self.camera_connected = bool(self._cap and self._cap.isOpened())
        if not self.camera_connected:
            if not self.reconnect_on_read_failure:
                self._release()
                raise RuntimeError(f"failed to open video file: {self.source}")
            logger.warning("video source connection failed source=%s retry_in_seconds=%.1f", self.source, self.reconnect_delay_seconds)
            self._release()
            time.sleep(self.reconnect_delay_seconds)
        else:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = frame_count / fps if fps and fps > 0 else 0.0
            logger.info(
                "video source connected source=%s fps=%.2f frame_count=%s duration_seconds=%.2f live=%s",
                self.source,
                fps or 0.0,
                frame_count,
                duration_seconds,
                self.reconnect_on_read_failure,
            )

    def _release(self) -> None:
        """Release the current OpenCV capture handle if it exists.

        Returns:
            ``None``.
        """
        if self._cap is not None:
            self._cap.release()
        self._cap = None


def _normalize_source(source: str):
    """Convert source URI shortcuts into OpenCV-compatible values.

    Args:
        source: Source string from CLI or configuration.

    Returns:
        Integer webcam index for ``webcam://N``, file path for ``file://``, or
        the original source string for RTSP/plain file paths.
    """
    if source.startswith("webcam://"):
        return int(source.removeprefix("webcam://"))
    if source.startswith("file://"):
        return str(Path(source.removeprefix("file://")))
    return source


def _is_reconnectable_source(source) -> bool:
    """Return whether a source should reconnect after read failure.

    Args:
        source: Normalized source value passed to OpenCV.

    Returns:
        ``True`` for webcam indexes and external stream URIs, ``False`` for
        local file paths.
    """
    if isinstance(source, int):
        return True
    if not isinstance(source, str):
        return True
    parsed = urlparse(source)
    return bool(parsed.scheme and parsed.scheme not in {"file"})
