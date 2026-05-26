from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .config import AppConfig, ProcessingConfig
from .schemas import TrackedVehicle

PlateSkipReason = Literal["class", "static"]


@dataclass(frozen=True, slots=True)
class PlateDetectionDecision:
    """Decision describing whether a track should run plate detection.

    Input fields:
        should_detect: Whether to run the plate detector.
        skip_reason: Reason plate detection was skipped, if any.

    Output:
        Consumed by the frame pipeline to control inference cost and logging.
    """

    should_detect: bool
    skip_reason: PlateSkipReason | None = None

    def as_tuple(self) -> tuple[bool, PlateSkipReason | None]:
        """Return the decision in tuple form for compact tests and callers.

        Returns:
            ``(should_detect, skip_reason)``.
        """
        return self.should_detect, self.skip_reason


@dataclass(slots=True)
class FrameProcessingStats:
    """Counters collected while processing one sampled frame.

    Input fields:
        plate_attempts: Number of vehicle crops sent to plate detection.
        plate_detections: Number of plate detections returned.
        skipped_static: Tracks skipped because they did not move enough.
        skipped_class: Tracks skipped because their class is not eligible.

    Output:
        Provides compact frame-level logging.
    """

    plate_attempts: int = 0
    plate_detections: int = 0
    skipped_static: int = 0
    skipped_class: int = 0

    def record_skip(self, reason: PlateSkipReason | None) -> None:
        """Increment the counter matching a skip reason.

        Args:
            reason: Skip reason returned by ``PlateDetectionPolicy``.

        Returns:
            ``None``.
        """
        if reason == "class":
            self.skipped_class += 1
        elif reason == "static":
            self.skipped_static += 1

    def record_plate_attempt(self, detections_count: int) -> None:
        """Record one plate detector invocation.

        Args:
            detections_count: Number of plate detections returned.

        Returns:
            ``None``.
        """
        self.plate_attempts += 1
        self.plate_detections += detections_count


class PlateDetectionPolicy:
    """Rules that decide whether plate detection should run for a track.

    Input:
        Created with the runtime processing configuration.

    Output:
        Produces ``PlateDetectionDecision`` values for tracked vehicles.
    """

    def __init__(self, config: ProcessingConfig) -> None:
        """Create a policy from processing configuration.

        Args:
            config: Processing controls from ``AppConfig``.

        Returns:
            ``None``.
        """
        self.min_displacement_px = config.min_plate_track_displacement_px
        self.allowed_classes = set(config.plate_detection_classes)

    def evaluate(self, track: TrackedVehicle) -> PlateDetectionDecision:
        """Decide whether plate detection should run for a track.

        Args:
            track: Current tracked vehicle state.

        Returns:
            ``PlateDetectionDecision`` with a skip reason when detection should
            not run.
        """
        if track.class_name not in self.allowed_classes:
            return PlateDetectionDecision(False, "class")
        if self.track_displacement(track) < self.min_displacement_px:
            return PlateDetectionDecision(False, "static")
        return PlateDetectionDecision(True)

    @staticmethod
    def track_displacement(track: TrackedVehicle) -> float:
        """Compute centroid displacement from first to latest trajectory point.

        Args:
            track: Vehicle track with a trajectory.

        Returns:
            Euclidean pixel distance between the first and latest trajectory
            point.
        """
        if len(track.trajectory) < 2:
            return 0.0
        start = track.trajectory[0]
        end = track.trajectory[-1]
        return ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** 0.5


def should_detect_plate(track: TrackedVehicle, config: AppConfig | ProcessingConfig) -> tuple[bool, PlateSkipReason | None]:
    """Evaluate the plate-detection policy and return a tuple result.

    Args:
        track: Current tracked vehicle state.
        config: Full app config or the nested processing config.

    Returns:
        ``(True, None)`` when plate detection should run, otherwise ``False``
        with a skip reason of ``class`` or ``static``.
    """
    processing_config = config.processing if isinstance(config, AppConfig) else config
    return PlateDetectionPolicy(processing_config).evaluate(track).as_tuple()


def summarize_classes(class_names: Iterable[str]) -> dict[str, int]:
    """Count class names for compact frame-level logs.

    Args:
        class_names: Iterable of class label strings.

    Returns:
        Mapping from class label to count.
    """
    counts: dict[str, int] = {}
    for class_name in class_names:
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts
