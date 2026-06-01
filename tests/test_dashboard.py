import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from config import FrameScore, ManeuverResult, TrafficLightResult, TrafficLightState, MotionResult
from dashboard import Dashboard


def make_dashboard():
    """Return a Dashboard with a mocked asyncio loop."""
    loop = MagicMock()
    # run_coroutine_threadsafe should not crash when called
    loop.run_coroutine_threadsafe = MagicMock()
    d = Dashboard(loop=loop)
    return d


@pytest.fixture
def dashboard():
    return make_dashboard()


@pytest.fixture
def black_frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


# ─── _build_payload tests ─────────────────────────────────────────────────

def test_build_payload_with_all_values(dashboard, black_frame):
    fs = FrameScore(score=87.3, error_cm=1.8, error_px=17.1)
    mr = ManeuverResult(maneuver_name="straight_1", confidence=1.0, payload="x", bbox=None)
    tl = TrafficLightResult(state=TrafficLightState.GREEN, confidence=0.9, bbox=None)
    mo = MotionResult(is_moving=True, pixel_change_ratio=0.03, changed_pixels=30, total_pixels=1000)

    p = dashboard._build_payload(black_frame, fs, mr, tl, mo)

    assert p["score"] == 87.3
    assert p["error_cm"] == 1.8
    assert p["maneuver"] == "straight_1"
    assert p["traffic_light"] == "green"
    assert p["is_moving"] is True
    assert p["frame_b64"] is not None
    assert p["frame_b64"].startswith("data:image/jpeg;base64,")


def test_build_payload_all_none(dashboard, black_frame):
    p = dashboard._build_payload(black_frame, None, None, None, None)
    assert p["score"] is None
    assert p["error_cm"] is None
    assert p["maneuver"] is None
    assert p["traffic_light"] == "none"
    assert p["is_moving"] is False


def test_build_payload_red_light(dashboard, black_frame):
    tl = TrafficLightResult(state=TrafficLightState.RED, confidence=0.95, bbox=None)
    p = dashboard._build_payload(black_frame, None, None, tl, None)
    assert p["traffic_light"] == "red"


def test_build_payload_score_rounded(dashboard, black_frame):
    fs = FrameScore(score=83.456789, error_cm=2.345, error_px=22.3)
    p = dashboard._build_payload(black_frame, fs, None, None, None)
    assert p["score"] == 83.5   # rounded to 1dp
    assert p["error_cm"] == 2.3


def test_emit_update_does_not_raise(dashboard, black_frame):
    """emit_update must be exception-safe — never crash Thread B."""
    fs = FrameScore(score=75.0, error_cm=3.0, error_px=28.0)
    # Should not raise even if sio.emit fails
    with patch.object(dashboard, '_build_payload', return_value={}):
        dashboard.emit_update(black_frame, fs, None, None, None)


def test_emit_update_calls_threadsafe(dashboard, black_frame):
    """Verify asyncio.run_coroutine_threadsafe is called when emitting."""
    with patch('asyncio.run_coroutine_threadsafe') as mock_threadsafe:
        dashboard.emit_update(black_frame, None, None, None, None)
        assert mock_threadsafe.called, "run_coroutine_threadsafe should be called once per emission"
