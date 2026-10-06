from pathlib import Path
import numpy as np
from ultralytics import YOLO
from .contracts import Detection


class YoloDetector:
    """Detect people. Tracking is handled separately."""

    def __init__(self, weights: str, device: str = "cpu", imgsz: int = 640):
        self.model = YOLO(weights)
        self.weights = str(Path(weights))
        self.device = device
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # ByteTrack also uses low-confidence detections.
        result = self.model.predict(frame, device=self.device, imgsz=self.imgsz,
                                    conf=0.1, classes=[0], verbose=False)[0]
        return [Detection(tuple(map(float, row[:4])), float(row[4]), int(row[5]))
                for row in result.boxes.data.cpu().numpy()]
