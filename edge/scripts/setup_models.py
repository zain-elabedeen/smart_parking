from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

SCRIPT_DIR = Path(__file__).resolve().parent
EDGE_DIR = SCRIPT_DIR.parent
if str(EDGE_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_DIR))

from app.config import AppConfig, load_config


ULTRALYTICS_DETECTION_MODELS = {
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
}


def main() -> int:
    """Download or validate the YOLO model files configured for the demo.

    Input:
        Reads CLI arguments and the YAML config file.

    Output:
        Writes missing vehicle models to disk, optionally downloads a plate
        model from a supplied URL, and returns a process exit code.
    """
    parser = argparse.ArgumentParser(description="Set up YOLO model files for the parking edge demo")
    parser.add_argument("--config", default="edge/config.yaml", help="Path to edge config YAML")
    parser.add_argument("--plate-model-url", default=None, help="Optional URL to download the plate detector .pt file")
    parser.add_argument("--skip-plate", action="store_true", help="Only set up the vehicle detector model")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_vehicle_model(config)

    if args.skip_plate:
        print("Skipped plate detector setup.")
        return 0

    plate_url = args.plate_model_url or config.plate_detector.model_url
    if ensure_plate_model(config, plate_url):
        return 0

    print(
        "Plate detector model is missing. Add edge/models/license_plate_detector.pt "
        "or rerun with --plate-model-url <url-to-yolo-plate-model.pt>.",
        file=sys.stderr,
    )
    return 2


def ensure_vehicle_model(config: AppConfig) -> Path:
    """Ensure the configured vehicle detector model exists.

    Args:
        config: Validated application configuration.

    Returns:
        Path to the available vehicle model file.
    """
    model_path = Path(config.vehicle_detector.model_path)
    if model_path.exists():
        print(f"Vehicle model already exists: {model_path}")
        return model_path

    model_name = model_path.name
    if model_name not in ULTRALYTICS_DETECTION_MODELS:
        raise RuntimeError(f"Vehicle model is missing and is not a known Ultralytics model name: {model_path}")

    download_ultralytics_model(model_name, model_path)
    print(f"Downloaded vehicle model: {model_path}")
    return model_path


def ensure_plate_model(config: AppConfig, model_url: str | None) -> Path | None:
    """Ensure the configured plate detector model exists.

    Args:
        config: Validated application configuration.
        model_url: Optional URL to download the plate model when missing.

    Returns:
        Path to the available plate model, or ``None`` if it is missing and no
        URL was provided.
    """
    model_path = Path(config.plate_detector.model_path)
    if model_path.exists():
        print(f"Plate model already exists: {model_path}")
        return model_path

    if not model_url:
        return None

    download_file(model_url, model_path)
    print(f"Downloaded plate model: {model_path}")
    return model_path


def download_ultralytics_model(model_name: str, destination: Path) -> None:
    """Download an official Ultralytics model through the YOLO API.

    Args:
        model_name: Official model filename, such as ``yolov8s.pt``.
        destination: Final local path where the model should be copied.

    Returns:
        ``None``.
    """
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("ultralytics is required to download official YOLO models") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)
            model = YOLO(model_name)
            source = Path(getattr(model, "ckpt_path", model_name))
            if not source.exists():
                source = Path(model_name)
            if not source.exists():
                raise RuntimeError(f"Ultralytics did not create expected model file: {model_name}")
            shutil.copy2(source, destination)
        finally:
            os.chdir(original_cwd)


def download_file(url: str, destination: Path) -> None:
    """Download a file from a URL to a local destination.

    Args:
        url: Source URL.
        destination: Local output path.

    Returns:
        ``None``.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        urlretrieve(url, temp_path)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())

