import pytest
from video_tracking.contracts import Detection, FrameKey, Track
from video_tracking.tracking import ByteTrackAdapter
from video_tracking.analytics import LineCounter


def key(seq, session='session-a'):
    return FrameKey('video-0', session, seq)


def test_id_survives_motion_low_confidence_and_short_missing_observation():
    tracker = ByteTrackAdapter(30)
    def detect(x, score=0.9):
        return [Detection((x, 10, x+40, 100), score, 0)]
    first = tracker.update(key(0), detect(10), (200, 200))
    assert len(first) == 1
    second = tracker.update(key(1), detect(12, 0.2), (200, 200))
    assert second[0].track_id == first[0].track_id
    assert tracker.update(key(2), [], (200, 200)) == []
    recovered = tracker.update(key(3), detect(16), (200, 200))
    assert recovered[0].track_id == first[0].track_id


def test_order_and_session_guards_do_not_advance_tracker():
    tracker = ByteTrackAdapter(30)
    tracker.update(key(0), [], (100, 100))
    for invalid in [key(0), key(2), key(1, 'other')]:
        with pytest.raises(ValueError):
            tracker.update(invalid, [], (100, 100))
    assert tracker.update(key(1), [], (100, 100)) == []


def test_crossings_deadband_and_expiry():
    counter = LineCounter(50, dead_band=5, ttl_frames=3)
    def observe(y, seq):
        counter.update([Track((0, y-5, 10, y+5), 1, 0.9, 0)], seq)
    for seq, y in enumerate([40, 49, 51, 60, 61, 40]):
        observe(y, seq)
    assert (counter.up, counter.down) == (1, 1)
    observe(60, 20)
    assert (counter.up, counter.down) == (1, 1)
