"""Unit tests for boundary snapping and padding enforcement."""
import pytest
from clipper.analyze import snap_boundaries


def test_snap_to_silence_edges():
    """Verify boundaries snap to nearest silence edges and add padding."""
    # Video duration = 60s
    # Silence from 9.0s to 11.0s, and 29.0s to 31.0s
    silences = [[9.0, 11.0], [29.0, 31.0]]
    scene_cuts = []

    # Raw window: [10.0, 30.0] -> start is inside silence [9, 11], end inside silence [29, 31]
    # start snaps to silence end (11.0), end snaps to silence start (29.0)
    # with 0.3s padding: start = 11.0 - 0.3 = 10.7, end = 29.0 + 0.3 = 29.3
    s_start, s_end = snap_boundaries(
        raw_start=10.0,
        raw_end=30.0,
        total_duration=60.0,
        scene_cuts=scene_cuts,
        silences=silences,
        snap_window=2.5,
        padding=0.3,
        min_length=15.0,
        max_length=40.0,
    )
    assert s_start == pytest.approx(10.7, abs=0.1)
    assert s_end == pytest.approx(29.3, abs=0.1)
    assert s_end - s_start >= 15.0


def test_snap_to_scene_cuts():
    """Verify boundaries snap to nearest scene cut within snap window."""
    silences = []
    scene_cuts = [14.8, 35.2]

    # Raw window: [15.0, 35.0]
    # start snaps to 14.8, end snaps to 35.2
    # with 0.3 padding: start = 14.8 - 0.3 = 14.5, end = 35.2 + 0.3 = 35.5
    s_start, s_end = snap_boundaries(
        raw_start=15.0,
        raw_end=35.0,
        total_duration=60.0,
        scene_cuts=scene_cuts,
        silences=silences,
        snap_window=2.0,
        padding=0.3,
        min_length=15.0,
        max_length=40.0,
    )
    assert s_start == pytest.approx(14.5, abs=0.1)
    assert s_end == pytest.approx(35.5, abs=0.1)


def test_clamped_boundaries_at_video_edges():
    """Verify padding never goes below 0.0 or above total duration."""
    s_start, s_end = snap_boundaries(
        raw_start=0.1,
        raw_end=20.0,
        total_duration=20.0,
        scene_cuts=[],
        silences=[],
        padding=0.5,
        min_length=10.0,
        max_length=30.0,
    )
    assert s_start >= 0.0
    assert s_end <= 20.0

