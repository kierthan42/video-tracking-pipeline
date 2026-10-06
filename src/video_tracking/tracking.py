from types import SimpleNamespace
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from .contracts import Detection, FrameKey, Track


class ByteTrackAdapter:
    """One tracker per session; frames must arrive in order without gaps."""

    def __init__(self, fps: float):
        self._tracker = BYTETracker(SimpleNamespace(
            track_high_thresh=0.25, track_low_thresh=0.1,
            new_track_thresh=0.25, track_buffer=30,
            match_thresh=0.8, fuse_score=True), frame_rate=round(fps))
        self._session = None
        self._next_seq = 0

    def update(self, key: FrameKey, detections: list[Detection],
               shape: tuple[int, int]) -> list[Track]:
        session = (key.stream_id, key.session_id)
        if self._session is not None and session != self._session:
            raise ValueError("Create a new tracker for a new session")
        if key.seq != self._next_seq:
            raise ValueError(f"Expected sequence {self._next_seq}, got {key.seq}")
        rows = np.asarray([(*d.xyxy, d.score, d.class_id) for d in detections],
                          dtype=np.float32).reshape(-1, 6)
        tracks = self._tracker.update(Boxes(rows, orig_shape=shape))
        self._session = session
        self._next_seq += 1
        return [Track(tuple(map(float, t[:4])), int(t[4]), float(t[5]), int(t[6]))
                for t in tracks]
