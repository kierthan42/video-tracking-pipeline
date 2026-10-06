from .contracts import Track


class LineCounter:
    """Count line crossings, including return trips. Forget old IDs after a while."""
    def __init__(self, y: float, dead_band: float = 5, ttl_frames: int = 60):
        self.y, self.dead_band, self.ttl = y, dead_band, ttl_frames
        self.up = self.down = 0
        self._seen = {}

    def update(self, tracks: list[Track], seq: int):
        self._seen = {k: v for k, v in self._seen.items() if seq - v[1] <= self.ttl}
        for t in tracks:
            if t.class_id != 0:
                continue
            delta = (t.xyxy[1] + t.xyxy[3]) / 2 - self.y
            side = 1 if delta > self.dead_band else -1 if delta < -self.dead_band else 0
            previous, _ = self._seen.get(t.track_id, (0, seq))
            if side and previous and side != previous:
                self.down += int(side == 1)
                self.up += int(side == -1)
            self._seen[t.track_id] = (side or previous, seq)
