"""Unit tests for signal processing and normalization."""
import numpy as np
import pytest
from clipper.signals import extract_heatmap_signal, normalize_signal


def test_normalize_signal_range():
    """Verify normalize_signal scales strictly to [0.0, 1.0]."""
    raw = np.array([10.0, 20.0, 50.0, 100.0], dtype=np.float32)
    norm = normalize_signal(raw)
    assert norm[0] == pytest.approx(0.0)
    assert norm[-1] == pytest.approx(1.0)
    assert np.all(norm >= 0.0) and np.all(norm <= 1.0)


def test_normalize_constant_signal():
    """Verify constant signal normalizes to zeros without division by zero."""
    constant = np.array([42.0, 42.0, 42.0], dtype=np.float32)
    norm = normalize_signal(constant)
    assert np.all(norm == 0.0)


def test_heatmap_interpolation():
    """Verify interpolation of sparse heatmap points to 1-second grid."""
    heatmap_data = [
        {"start_time": 0.0, "end_time": 10.0, "value": 0.1},
        {"start_time": 10.0, "end_time": 20.0, "value": 0.9},
        {"start_time": 20.0, "end_time": 30.0, "value": 0.2},
    ]
    total_seconds = 30
    signal, has_heat = extract_heatmap_signal(heatmap_data, total_seconds)
    assert has_heat is True
    assert len(signal) == 30
    # Peak should be near index 15
    assert np.argmax(signal) == 15
    assert signal[15] == pytest.approx(1.0)

