# Smart Parking Edge Node

Prototype edge computer-vision pipeline for vehicle detection, tracking, license-plate crop selection, direction estimation, and durable event publishing.

## What is included

- Python edge-node package under `edge/app`.
- YAML configuration at `edge/config.yaml`.
- Local SQLite outbound queue with idempotent event inserts and retry backoff.
- Deterministic line-crossing direction estimator.
- Per-track license-plate crop quality selector.
- Pluggable YOLO vehicle and plate detectors.
- ByteTrack vehicle tracker through `supervision`.
- Local object-storage adapter for plate crop JPEGs.
- Queue-backed Kafka publisher.
- FastAPI health app factory and Prometheus metric definitions.
- Unit tests for direction, crop scoring, and queue behavior.

## Setup

```bash
cd edge
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Set up models:

```bash
python scripts/setup_models.py --config config.yaml --skip-plate
```

That downloads the configured COCO vehicle model (`edge/models/yolov8s.pt`) and the configured Hugging Face license-plate model (`edge/models/license_plate_detector.pt`).

```bash
python scripts/setup_models.py --config config.yaml
```

The configured plate model is `Koushim/yolov8-license-plate-detection` from Hugging Face. It is a YOLOv8n detector with one class, `license_plate`, and the repository publishes the weight file as `best.pt`.

Add demo videos under `edge/demo_videos/`.

## Run

From the repository root:

```bash
python edge/app/main.py --config edge/config.yaml --source edge/demo_videos/entry.mp4 --no-kafka
```

## Docker

Build from the repository root:

```bash
docker build -t smart-parking-edge:latest edge
```

Run from `edge/`, mounting videos and runtime data:

```bash
docker run --rm \
  -p 9101:9101 \
  -v "$PWD/demo_videos:/app/edge/demo_videos:ro" \
  -v "$PWD/data:/app/edge/data" \
  smart-parking-edge:latest
```

The default Docker command runs with `--no-kafka` so the container works for local demos without a Kafka broker. Events are kept in the mounted SQLite queue under `edge/data/`.

## Test

```bash
python3 -m pytest edge/tests
```

## Notes

- Relative paths in `edge/config.yaml` are resolved from the config file directory, so `models/yolov8s.pt` means `edge/models/yolov8s.pt`.
- `--no-kafka` keeps events in the local SQLite queue without requiring a broker.
- The vehicle detector uses COCO classes (`car`, `truck`, `bus`, `motorcycle`) and the tracker assigns stable ByteTrack ids.
