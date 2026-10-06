# Real-Time Video Tracking Pipeline

A video tracking project for learning computer vision and distributed processing.

Right now it takes a recorded video, detects people with YOLO, tracks them with
ByteTrack, and saves an annotated video. It also counts crossings over a horizontal
line and logs timing stats. Detection and tracking are separate so detection can
move into Redis workers later.

## Setup

Use Python 3.12. Python 3.9 won't work.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

If you already have a `.venv` made with Python 3.9, create a new environment with
Python 3.12 instead. Activating the old one won't change its Python version.

## Run

```bash
video-track video.mp4 --output-dir runs/test --device cpu
```

The default model is `yolo11n.pt`; it downloads on the first run. The output
directory needs to be new each time.

For a shorter run with a smaller inference size:

```bash
video-track video.mp4 --output-dir runs/test-320 --imgsz 320 --max-frames 300
```

You can pass `--weights path/to/yolo11n.pt` to use a local model. CPU is the default;
`--device mps` or `--device 0` can be used with a compatible GPU setup.

The output folder contains:

- `annotated.mp4` — video with boxes, IDs and crossing counts
- `tracks.jsonl` — tracks for each frame
- `frames.csv` — per-frame timings
- `metrics.json` — FPS and timing summary

This currently expects a constant-frame-rate recording. Only people are detected.
IDs can change after occlusions, and crossing counts aren't unique-person counts.
Processing FPS is measured offline, not live camera-to-screen performance.

## Tests

```bash
pytest -q
```

Tests cover tracking, counting and video output. They don't download YOLO weights.
`scripts/make_fixture.py` can turn an image into a short video for a quick smoke test.

## Next

- Bounded queues and ordering for delayed detection results
- Redis workers
- Docker Compose and a basic live video view
- Try EC2 once the local version works

Model and library licensing: [Ultralytics](https://www.ultralytics.com/license).
