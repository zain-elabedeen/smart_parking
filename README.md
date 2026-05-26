# smart_parking

Edge computer-vision prototype for a parking entrance/exit camera. The app reads a video source, detects and tracks vehicles, detects license plates inside vehicle crops, keeps the best plate crop per track, estimates entry/exit direction, and writes durable `PlateCropDetected` events locally before optional Kafka publishing.

This project is designed as a Senior Software Engineer interview demo: small enough to run locally, but structured around the same boundaries an edge production system would need.

## What Is Included

- OpenCV video ingestion for MP4 files, webcams, and RTSP streams.
- Ultralytics YOLO vehicle detection using COCO vehicle classes.
- ByteTrack vehicle tracking through `supervision`.
- YOLO license-plate detection using the configured Hugging Face model.
- Per-track crop quality scoring and best-crop selection.
- Virtual-line direction estimation for `ENTRY`, `EXIT`, or `UNKNOWN`.
- SQLite outbound queue with idempotent event IDs and retry state.
- Local crop storage under `edge/data/crops`.
- Optional Kafka publishing through `confluent-kafka`.
- Prometheus metrics server and FastAPI health server support.
- Dockerfile and model setup script.
- Unit tests for config, direction, crop scoring, queueing, processing policy, tracking integration, and video EOF behavior.

## Repository Layout

```text
smart_parking/
  README.md
  edge/
    app/
      main.py              # CLI entrypoint
      bootstrap.py         # Runtime wiring
      pipeline.py          # Frame processing orchestration
      event_service.py     # Event/crop persistence
      processing.py        # Plate-detection policy and frame stats
      *_detector.py        # YOLO adapters
      vehicle_tracker.py   # ByteTrack adapter
      local_queue.py       # SQLite durable queue
    config.yaml
    Dockerfile
    requirements.txt
    scripts/setup_models.py
    models/README.md
    tests/
```

## Prerequisites

- Python 3.11 or 3.12.
- `pip` and a virtual environment.
- A local MP4 demo video, webcam, or RTSP camera.
- Network access only when downloading models or installing dependencies.
- Docker, optional.

## Local Setup

From the repository root:

```bash
cd edge
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the configured models:

```bash
python scripts/setup_models.py --config config.yaml
```

The script downloads:

- `models/yolov8s.pt` for vehicle detection, if missing.
- `models/license_plate_detector.pt` from `Koushim/yolov8-license-plate-detection`, if missing.

To download only the vehicle model and provide the plate model manually later:

```bash
python scripts/setup_models.py --config config.yaml --skip-plate
```

Large runtime artifacts are intentionally ignored by Git: model weights, demo videos, crop images, SQLite data, caches, and bytecode.

## Running The Demo

Put videos under `edge/demo_videos/`, then run from `edge/`:

```bash
python -m app.main --config config.yaml --source demo_videos/entry.mp4 --no-kafka
```

Useful options:

```bash
python -m app.main --config config.yaml --source demo_videos/entry.mp4 --no-kafka --max-frames 100
python -m app.main --config config.yaml --source webcam://0 --no-kafka
python -m app.main --config config.yaml --source rtsp://camera-ip/live
python -m app.main --config config.yaml --source demo_videos/entry.mp4 --no-kafka --verbose-tracks
```

Notes:

- Local MP4 files stop at end-of-file. Live sources such as RTSP and webcam reconnect on read failure.
- Video duration is not wall-clock runtime. Inference can take longer than playback, especially on CPU.
- `--no-kafka` keeps events in SQLite without requiring a Kafka broker.
- `--verbose-tracks` prints per-track crop and zero-detection logs; normal mode keeps logs compact.

## Outputs

Local crop images are saved when a `PlateCropDetected` event is published:

```text
edge/data/crops/plate-crops/<site_id>/<camera_id>/<event_id>.jpg
```

For the default config:

```bash
open data/crops/plate-crops/berlin-demo-site/entry-camera-1
find data/crops -name "*.jpg"
```

Queued events are stored in:

```text
edge/data/outbound_events.db
```

The event payload contains the crop object key instead of embedding base64 image data in Kafka.

## Metrics And Health

The main process starts a Prometheus metrics endpoint on the configured metrics port, default `9101`. In restricted local environments, binding this port may fail; the app logs a warning and continues processing.

The health API can be run separately from `edge/`:

```bash
python -m app.health --config config.yaml
```

It listens on `metrics.port + 1`, so the default health URL is:

```text
http://localhost:9102/health
```

## Docker

Build from the repository root after models are available under `edge/models/`:

```bash
docker build -t smart-parking-edge:latest edge
```

Run with local videos and runtime data mounted:

```bash
cd edge
docker run --rm \
  -p 9101:9101 \
  -v "$PWD/demo_videos:/app/edge/demo_videos:ro" \
  -v "$PWD/data:/app/edge/data" \
  smart-parking-edge:latest
```

The container default command uses `--no-kafka`. To use Kafka, override the command and point `kafka.bootstrap_servers` in `config.yaml` at a reachable broker.

## Tests And Checks

From the repository root:

```bash
python -m pytest edge/tests
python -m ruff check edge/app edge/tests
python -m compileall edge/app edge/scripts edge/tests
```

Current expected test result:

```text
19 passed
```

## Configuration Notes

- Relative paths in `edge/config.yaml` resolve from the config file directory.
- `frame_sampling.target_fps` controls how often frames go to inference.
- `processing.plate_detection_classes` defaults to `car`, `truck`, and `bus`, so motorcycles are tracked but skipped for plate detection.
- `processing.min_plate_track_displacement_px` skips plate detection for parked or nearly static tracks.
- The direction estimator uses the configured virtual line and `entry_direction` to classify crossings.

## Troubleshooting

- `video file reached end`: normal for MP4 input; the app stops at EOF.
- Runtime is longer than the video duration: expected when inference is slower than real-time.
- No crops saved: a crop is saved only when the pipeline publishes a `PlateCropDetected` event. Check logs for `saved crop` or use `--verbose-tracks` to inspect detections.
- Metrics port warning: another process or the sandbox blocked the port; processing continues.
- Matplotlib/font cache warnings from dependencies are usually harmless. Set `MPLCONFIGDIR` to a writable directory if startup is slow because of font-cache rebuilds.
