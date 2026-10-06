import csv
import json
import math
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import cv2

from .analytics import LineCounter
from .contracts import Detector, FrameKey, Tracker
from .metrics import Measurements


def annotate(frame, tracks, counter, seq):
    output = frame.copy()
    cv2.line(output, (0, int(counter.y)), (output.shape[1] - 1, int(counter.y)), (0, 255, 255), 2)
    for track in tracks:
        x1, y1, x2, y2 = map(round, track.xyxy)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(output, f"person #{track.track_id} {track.score:.2f}",
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    cv2.putText(output, f"frame {seq} | up {counter.up} down {counter.down}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return output


def run_video(source: Path, output_dir: Path, detector: Detector, tracker_factory,
              max_frames: int | None = None, fps_override: float | None = None,
              run_config: dict | None = None):
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    if not source.is_file():
        raise ValueError(f"Input video does not exist: {source}")
    # Keep previous runs from being overwritten.
    output_dir.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(source))
    writer = None
    session = str(uuid4())
    measurements = Measurements()
    tracks_total = 0
    frame_count = 0
    started = time.perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        fps = fps_override if fps_override is not None else capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has invalid FPS; specify --fps")
        tracker: Tracker = tracker_factory(fps)
        counter = None
        size = None
        with (output_dir / "frames.csv").open("w", newline="") as timings, \
             (output_dir / "tracks.jsonl").open("w") as track_file:
            columns = ["seq", "source_time_ms", "ingested_unix_ns", "decode_ms",
                       "detect_ms", "track_ms", "render_write_ms", "processing_latency_ms"]
            csv_writer = csv.DictWriter(timings, fieldnames=columns)
            csv_writer.writeheader()
            while max_frames is None or frame_count < max_frames:
                t0 = time.perf_counter()
                ok, frame = capture.read()
                t1 = time.perf_counter()
                ingested = time.time_ns()
                if not ok:
                    break
                h, w = frame.shape[:2]
                if writer is None:
                    size = (w, h)
                    counter = LineCounter(h / 2, ttl_frames=max(1, round(fps * 2)))
                    writer = cv2.VideoWriter(str(output_dir / "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                    if not writer.isOpened():
                        raise ValueError("Cannot open mp4v video writer")
                elif size != (w, h):
                    raise ValueError("Frame dimensions changed during playback")
                key = FrameKey("video-0", session, frame_count)
                detect_started = time.perf_counter()
                detections = detector.detect(frame)
                t2 = time.perf_counter()
                tracks = tracker.update(key, detections, (h, w))
                counter.update(tracks, frame_count)
                t3 = time.perf_counter()
                writer.write(annotate(frame, tracks, counter, frame_count))
                t4 = time.perf_counter()
                values = dict(decode_ms=(t1-t0)*1000, detect_ms=(t2-detect_started)*1000,
                              track_ms=(t3-t2)*1000, render_write_ms=(t4-t3)*1000,
                              processing_latency_ms=(t4-t0)*1000)
                measurements.add(values)
                # Media time, assuming constant FPS.
                csv_writer.writerow(dict(seq=frame_count, source_time_ms=frame_count/fps*1000,
                                         ingested_unix_ns=ingested, **values))
                track_file.write(json.dumps(dict(**asdict(key),
                    tracks=[asdict(t) for t in tracks])) + "\n")
                tracks_total += len(tracks)
                frame_count += 1
        if frame_count == 0:
            raise ValueError("Video contained no decodable frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    elapsed = time.perf_counter() - started
    summary = dict(schema_version=1, session_id=session, frames=frame_count,
                   fps_source=fps, resolution=list(size), elapsed_seconds=elapsed,
                   processing_fps=frame_count/elapsed, source_seconds=frame_count/fps,
                   observed_track_boxes=tracks_total,
                   counts=dict(up=counter.up, down=counter.down),
                   dropped_frames=0, queue_depth_max=0,
                   timing_ms=measurements.summary(), percentile_samples=len(measurements.samples),
                   mode="offline_lossless", config=run_config or {},
                   environment=dict(python=platform.python_version(), platform=platform.platform(),
                                    machine=platform.machine(), opencv=cv2.__version__,
                                    ultralytics=version("ultralytics"), torch=version("torch")))
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
