# Models

Place model files here or mount them into this directory at runtime.

Expected MVP files:

- `yolov8s.pt` or another COCO vehicle detector compatible with Ultralytics YOLO.
- `license_plate_detector.pt`, a YOLO plate detector.

The repository intentionally does not include large model binaries.

Run this from `edge/` to download the configured vehicle and plate models:

```bash
python scripts/setup_models.py --config config.yaml
```

The configured plate detector is downloaded from `Koushim/yolov8-license-plate-detection` on Hugging Face and saved locally as `license_plate_detector.pt`.
