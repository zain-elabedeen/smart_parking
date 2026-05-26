from __future__ import annotations

import logging
from pathlib import Path

from .schemas import PlateDetection

logger = logging.getLogger(__name__)


class PlateDetector:
    """YOLO-backed license plate detector for vehicle crops.

    Input:
        Created with a plate model path, confidence threshold, and minimum crop
        dimensions.

    Output:
        Produces ``PlateDetection`` objects from vehicle crop images.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.40,
        min_plate_width_px: int = 40,
        min_plate_height_px: int = 12,
    ) -> None:
        """Load a YOLO license-plate detector model.

        Args:
            model_path: Path to an Ultralytics-compatible plate model.
            confidence_threshold: Minimum plate confidence accepted.
            min_plate_width_px: Minimum detected plate width in pixels.
            min_plate_height_px: Minimum detected plate height in pixels.

        Returns:
            ``None``.
        """
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("ultralytics is required for YOLO plate detection") from exc

        if not Path(model_path).exists():
            raise FileNotFoundError(f"plate detector model not found at {model_path}; provide a YOLO plate model first")

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.min_plate_width_px = min_plate_width_px
        self.min_plate_height_px = min_plate_height_px
        logger.info(
            "loaded plate YOLO model path=%s confidence_threshold=%.2f min_size=%sx%s",
            model_path,
            confidence_threshold,
            min_plate_width_px,
            min_plate_height_px,
        )

    def detect(self, vehicle_crop) -> list[PlateDetection]:
        """Detect license plates inside one vehicle crop.

        Args:
            vehicle_crop: OpenCV image array cropped to a vehicle bounding box.

        Returns:
            List of plate detections with boxes relative to ``vehicle_crop``.
        """
        if vehicle_crop.size == 0:
            return []

        results = self.model(vehicle_crop, conf=self.confidence_threshold, verbose=False)
        detections: list[PlateDetection] = []

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf.item())
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)

                if width < self.min_plate_width_px or height < self.min_plate_height_px:
                    continue
                
                crop = vehicle_crop[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                detections.append(PlateDetection(bbox=[float(x1), float(y1), float(x2), float(y2)], confidence=confidence, crop=crop))
        return detections
