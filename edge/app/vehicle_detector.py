from __future__ import annotations

import logging
from pathlib import Path

from .schemas import Detection, Frame


COCO_VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle"}
logger = logging.getLogger(__name__)


class VehicleDetector:
    """YOLO-backed vehicle detector.

    Input:
        Created with a YOLO model path, confidence threshold, and optional
        class allow-list.

    Output:
        Produces vehicle ``Detection`` objects for each frame.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.45, classes: list[str] | None = None) -> None:
        """Load a YOLO vehicle detector model.

        Args:
            model_path: Path to an Ultralytics-compatible model.
            confidence_threshold: Minimum confidence accepted.
            classes: Optional accepted class names; defaults to COCO vehicles.

        Returns:
            ``None``.
        """
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("ultralytics is required for YOLO vehicle detection") from exc

        if not Path(model_path).exists():
            raise FileNotFoundError(f"vehicle model not found at {model_path}; run edge/scripts/setup_models.py first")

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.allowed_classes = set(classes or COCO_VEHICLE_CLASS_NAMES)
        logger.info("loaded vehicle YOLO model path=%s allowed_classes=%s", model_path, sorted(self.allowed_classes))

    def detect(self, frame: Frame) -> list[Detection]:
        """Detect vehicles in one frame.

        Args:
            frame: Frame containing an OpenCV image array.

        Returns:
            List of accepted vehicle detections in full-frame coordinates.
        """
        results = self.model.predict(frame.image, conf=self.confidence_threshold, verbose=False)
        detections: list[Detection] = []

        for result in results:
            names = result.names

            for box in result.boxes:
                class_name = names[int(box.cls.item())]
                confidence = float(box.conf.item())
                
                if class_name not in self.allowed_classes or confidence < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(Detection(bbox=[x1, y1, x2, y2], class_name=class_name, confidence=confidence))
        return detections
