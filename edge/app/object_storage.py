from __future__ import annotations

import logging
from pathlib import Path

try:
    import cv2
except Exception:  # pragma: no cover - runtime dependency
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)


class LocalObjectStorage:
    """Local filesystem implementation for storing plate crop images.

    Input:
        Created with a root directory.

    Output:
        Writes JPEG files and returns object keys for event payloads.
    """

    def __init__(self, root_path: str | Path) -> None:
        """Create the local storage root if needed.

        Args:
            root_path: Directory where object keys are resolved.

        Returns:
            ``None``.
        """
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    def save_jpeg(self, object_key: str, image, jpeg_quality: int = 90) -> str:
        """Save an image as a JPEG under the storage root.

        Args:
            object_key: Relative object key/path to write.
            image: OpenCV image array.
            jpeg_quality: JPEG quality from 0 to 100.

        Returns:
            The same object key that was written.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV is required to write crop images")

        path = self.root_path / object_key
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            raise RuntimeError(f"failed to write crop image: {path}")
        logger.info("wrote crop image path=%s object_key=%s", path, object_key)
        return object_key
