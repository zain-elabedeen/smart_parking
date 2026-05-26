from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.crop_selector import PlateCropCandidate, PlateCropSelector, laplacian_variance, score_candidate


def _candidate(crop, confidence: float, plate_bbox: list[float]) -> PlateCropCandidate:
    """Build a plate crop candidate fixture for crop selector tests.

    Args:
        crop: Image array to attach to the candidate.
        confidence: Plate detection confidence.
        plate_bbox: Plate box as ``[x1, y1, x2, y2]``.

    Returns:
        ``PlateCropCandidate`` with a fixed track and vehicle box.
    """
    return PlateCropCandidate(
        track_id=1,
        frame_id="frame-000001",
        timestamp=datetime.now(tz=timezone.utc),
        plate_bbox=plate_bbox,
        vehicle_bbox=[0, 0, 300, 200],
        crop=crop,
        detection_confidence=confidence,
    )


def test_laplacian_variance_prefers_sharp_edges() -> None:
    """Verify the sharpness metric is higher for high-frequency edges.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    blurry = np.full((32, 96), 120, dtype=np.uint8)
    sharp = np.zeros((32, 96), dtype=np.uint8)
    sharp[:, ::2] = 255

    assert laplacian_variance(sharp) > laplacian_variance(blurry)


def test_quality_score_uses_confidence_size_and_sharpness() -> None:
    """Verify stronger candidate attributes produce a higher quality score.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    blurry = np.full((32, 96), 120, dtype=np.uint8)
    sharp = np.zeros((48, 160), dtype=np.uint8)
    sharp[:, ::2] = 255

    weak = _candidate(blurry, 0.45, [100, 140, 180, 165])
    strong = _candidate(sharp, 0.95, [70, 130, 230, 178])

    assert score_candidate(strong) > score_candidate(weak)


def test_selector_keeps_best_candidate_per_track() -> None:
    """Verify the selector tracks the highest-scoring crop per vehicle.

    Returns:
        ``None``. Pytest assertions validate the behavior.
    """
    selector = PlateCropSelector()
    weak_crop = np.full((32, 96), 120, dtype=np.uint8)
    strong_crop = np.zeros((48, 160), dtype=np.uint8)
    strong_crop[:, ::2] = 255

    selector.add_candidate(_candidate(weak_crop, 0.5, [100, 140, 180, 165]))
    best = selector.add_candidate(_candidate(strong_crop, 0.95, [70, 130, 230, 178]))

    assert selector.best_for_track(1) is best
