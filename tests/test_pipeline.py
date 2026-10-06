import json
import cv2
import numpy as np
import pytest
from video_tracking.contracts import Detection
from video_tracking.pipeline import run_video
from video_tracking.tracking import ByteTrackAdapter
from video_tracking.metrics import Measurements


class TestDetector:
    __test__ = False
    def detect(self, frame):
        return [Detection((20, 20, 60, 80), 0.9, 0)]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'input.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
    assert writer.isOpened()
    for i in range(5):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (20, 20), (60, 80), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def test_video_output_identity_and_metrics(video, tmp_path):
    out = tmp_path / 'run'
    summary = run_video(video, out, TestDetector(), ByteTrackAdapter)
    assert summary['frames'] == 5
    assert summary['processing_fps'] > 0
    assert summary['queue_depth_max'] == summary['dropped_frames'] == 0
    rows = [json.loads(line) for line in (out / 'tracks.jsonl').read_text().splitlines()]
    assert [row['seq'] for row in rows] == list(range(5))
    assert len({row['tracks'][0]['track_id'] for row in rows}) == 1
    cap = cv2.VideoCapture(str(out / 'annotated.mp4'))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (120, 160, 3)
        decoded += 1
    cap.release()
    assert decoded == 5
    assert len((out / 'frames.csv').read_text().splitlines()) == 6
    with pytest.raises(FileExistsError):
        run_video(video, out, TestDetector(), ByteTrackAdapter)


def test_frame_limit_and_bad_input(video, tmp_path):
    summary = run_video(video, tmp_path/'limited', TestDetector(), ByteTrackAdapter, max_frames=2)
    assert summary['frames'] == 2
    with pytest.raises(ValueError, match='does not exist'):
        run_video(tmp_path/'missing.mp4', tmp_path/'bad', TestDetector(), ByteTrackAdapter)
    with pytest.raises(ValueError, match='invalid FPS'):
        run_video(video, tmp_path/'bad-fps', TestDetector(), ByteTrackAdapter, fps_override=0)


def test_metrics_memory_is_bounded():
    metrics = Measurements(capacity=10)
    for i in range(1000):
        metrics.add({'latency': float(i)})
    assert len(metrics.samples) == 10
    assert metrics.summary()['latency']['mean'] == 499.5
