from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class FrameKey:
    stream_id: str
    session_id: str
    seq: int


@dataclass(frozen=True)
class Detection:
    # Original frame pixels, not model letterbox coordinates.
    xyxy: tuple[float, float, float, float]
    score: float
    class_id: int


@dataclass(frozen=True)
class Track:
    xyxy: tuple[float, float, float, float]
    track_id: int
    score: float
    class_id: int


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class Tracker(Protocol):
    def update(self, key: FrameKey, detections: list[Detection],
               shape: tuple[int, int]) -> list[Track]: ...
