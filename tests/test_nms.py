"""Unit tests for Non-Maximum Suppression (NMS) and overlap removal."""
import pytest
from clipper.analyze import compute_iou


def test_iou_non_overlapping():
    """Verify IoU is zero for disjoint intervals."""
    assert compute_iou(0.0, 10.0, 15.0, 25.0) == 0.0
    assert compute_iou(10.0, 20.0, 20.0, 30.0) == 0.0


def test_iou_identical():
    """Verify IoU is 1.0 for identical intervals."""
    assert compute_iou(10.0, 30.0, 10.0, 30.0) == 1.0


def test_iou_partial_overlap():
    """Verify IoU calculation on partial overlap."""
    # [10, 30] (len 20) and [20, 40] (len 20)
    # intersection = [20, 30] (len 10)
    # union = 20 + 20 - 10 = 30
    # iou = 10 / 30 = 0.3333
    iou = compute_iou(10.0, 30.0, 20.0, 40.0)
    assert iou == pytest.approx(10.0 / 30.0, rel=1e-3)


def test_nms_selection_filters_overlap():
    """Verify high-scoring overlapping segments are suppressed in favor of top non-overlapping ones."""
    from clipper.analyze import detect_moments
    from clipper.download import VideoMetadata
    from clipper.signals import VideoSignals

    meta = VideoMetadata(
        job_id="test_nms",
        url_or_path="test",
        title="NMS Test",
        channel="Test",
        duration=100.0,
        width=1280,
        height=720,
        fps=30.0,
        chapters=[],
        heatmap=None,
        has_heatmap=False,
        source_file="test.mp4",
        audio_file="test.wav",
    )

    # Synthetic signals with two distinct high activity regions:
    # Region A: 10s - 30s
    # Region B: 60s - 80s
    audio_energy = [0.1] * 100
    for t in range(10, 30):
        audio_energy[t] = 0.95
    for t in range(60, 80):
        audio_energy[t] = 0.85

    signals = VideoSignals(
        duration=100.0,
        total_seconds=100,
        has_heatmap=False,
        heatmap=[0.0] * 100,
        audio_energy=audio_energy,
        visual_motion=[0.5] * 100,
        pacing=[0.9] * 100,
        scene_cuts=[],
        silences=[],
    )

    clips = detect_moments(meta, signals, num_clips=2, min_length=15.0, max_length=25.0)
    assert len(clips) == 2

    # Verify the two clips are non-overlapping or within NMS threshold
    c1, c2 = clips[0], clips[1]
    iou = compute_iou(c1.start_sec, c1.end_sec, c2.start_sec, c2.end_sec)
    assert iou <= 0.25

