"""Unit tests for moment scoring and weight redistribution."""
import pytest
from clipper.analyze import compute_effective_weights, generate_moment_reason, SignalBreakdown


def test_effective_weights_with_heatmap():
    """Verify weights normalize and keep heatmap when available."""
    config_weights = {
        "heatmap": 0.35,
        "audio": 0.30,
        "visual": 0.20,
        "pacing": 0.15,
    }
    weights = compute_effective_weights(has_heatmap=True, config_weights=config_weights)
    assert weights["heatmap"] == pytest.approx(0.35, rel=1e-2)
    assert weights["audio"] == pytest.approx(0.30, rel=1e-2)
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-2)


def test_effective_weights_without_heatmap():
    """Verify heatmap weight is redistributed proportionally when missing."""
    config_weights = {
        "heatmap": 0.35,
        "audio": 0.30,
        "visual": 0.20,
        "pacing": 0.15,
    }
    weights = compute_effective_weights(has_heatmap=False, config_weights=config_weights)
    assert weights["heatmap"] == 0.0
    # subtotal = 0.30 + 0.20 + 0.15 = 0.65
    assert weights["audio"] == pytest.approx(0.30 / 0.65, rel=1e-2)
    assert weights["visual"] == pytest.approx(0.20 / 0.65, rel=1e-2)
    assert weights["pacing"] == pytest.approx(0.15 / 0.65, rel=1e-2)
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-2)


def test_moment_reason_generation():
    """Verify dynamic one-line reason generation based on highest signals."""
    b1 = SignalBreakdown(heatmap=85.0, audio=70.0, visual=20.0, pacing=80.0)
    reason1 = generate_moment_reason(b1, has_heatmap=True)
    assert "Replay peak (85%)" in reason1
    assert "audio" in reason1.lower()

    b2 = SignalBreakdown(heatmap=0.0, audio=30.0, visual=65.0, pacing=75.0)
    reason2 = generate_moment_reason(b2, has_heatmap=False)
    assert "Replay peak" not in reason2
    assert "scene cuts" in reason2.lower()

