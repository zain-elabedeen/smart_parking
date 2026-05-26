from __future__ import annotations


def crop_image(image, bbox: list[float]):
    """Crop an image to a bounding box while clamping to image bounds.

    Args:
        image: OpenCV image array.
        bbox: Box as ``[x1, y1, x2, y2]``.

    Returns:
        Image array slice inside the requested box.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = min(width, max(0, x1))
    x2 = min(width, max(0, x2))
    y1 = min(height, max(0, y1))
    y2 = min(height, max(0, y2))
    return image[y1:y2, x1:x2]


def offset_bbox(bbox: list[float], offset_x: float, offset_y: float) -> list[float]:
    """Translate a crop-relative box into full-frame coordinates.

    Args:
        bbox: Box as ``[x1, y1, x2, y2]`` relative to a crop.
        offset_x: Horizontal crop offset in the full frame.
        offset_y: Vertical crop offset in the full frame.

    Returns:
        Translated box in full-frame coordinates.
    """
    return [bbox[0] + offset_x, bbox[1] + offset_y, bbox[2] + offset_x, bbox[3] + offset_y]


def round_bbox(bbox: list[float]) -> list[float]:
    """Round a bounding box for compact logging.

    Args:
        bbox: Box as ``[x1, y1, x2, y2]``.

    Returns:
        Box values rounded to one decimal place.
    """
    return [round(value, 1) for value in bbox]
