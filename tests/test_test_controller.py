import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from test_controller import TestController, TestState
from config import ManeuverScore, TestResult, ManeuverResult, FrameScore, MANEUVER_SEQUENCE

class DummyScoringEngine:
    def __init__(self):
        self.frame_scores = []
        self._penalty = 0.0
        self._violations = 0
        self._red_moving_streak = 0

    def _reset_buffer(self):
        self.frame_scores = []
        self._penalty = 0.0
        self._violations = 0
        self._red_moving_streak = 0

    def aggregate_maneuver(self, name: str) -> ManeuverScore:
        return ManeuverScore(
            name=name,
            raw_score=90.0,
            penalty=0.0,
            final_score=90.0,
            frame_count=10,
            violations=0
        )

@pytest.fixture
def test_controller():
    se = DummyScoringEngine()
    return TestController(se)

def test_initial_state(test_controller):
    assert test_controller.state == TestState.IDLE
    assert test_controller.current_maneuver == MANEUVER_SEQUENCE[0]

def test_start_test(test_controller):
    test_controller.start_test("cand_123")
    assert test_controller.state == TestState.RUNNING
    assert test_controller.candidate_id == "cand_123"

def test_start_while_running_raises(test_controller):
    test_controller.start_test("cand_123")
    with pytest.raises(RuntimeError):
        test_controller.start_test("cand_456")

def test_abort_resets_state(test_controller):
    test_controller.start_test("cand_123")
    test_controller.abort()
    assert test_controller.state == TestState.IDLE

def test_update_ignores_when_idle(test_controller):
    mr = ManeuverResult(maneuver_name="straight_1", confidence=1.0, payload="test", bbox=None)
    res = test_controller.update(maneuver_result=mr, frame_score=None)
    assert res is None
    assert test_controller._maneuver_index == 0 # no change

def test_update_processes_maneuvers(test_controller, monkeypatch):
    test_controller._persist = MagicMock()
    test_controller.start_test("cand_001")
    
    # 1. Start test, send first QR (close None, open straight_1)
    mr1 = ManeuverResult(maneuver_name="straight_1", confidence=1.0, payload="test", bbox=None)
    test_controller.update(maneuver_result=mr1, frame_score=None)
    assert test_controller.current_maneuver == "left_curve_1"
    assert len(test_controller.scores_so_far) == 1
    assert test_controller.scores_so_far[0].name == "straight_1"
    
    # Fast forward through rest of maneuvers
    for i in range(1, 6):
        mr = ManeuverResult(maneuver_name=MANEUVER_SEQUENCE[i], confidence=1.0, payload="test", bbox=None)
        test_controller.update(maneuver_result=mr, frame_score=None)
        
    assert len(test_controller.scores_so_far) == 6
    assert test_controller.state == TestState.RUNNING
    
    # Last maneuver transition finishes the test
    mr_last = ManeuverResult(maneuver_name=MANEUVER_SEQUENCE[6], confidence=1.0, payload="test", bbox=None)
    res = test_controller.update(maneuver_result=mr_last, frame_score=None)
    
    assert res is not None
    assert isinstance(res, TestResult)
    assert test_controller.state == TestState.FINISHED
    assert len(res.maneuvers) == 7
    assert res.candidate_id == "cand_001"
    
    # Using our mock scores (all 90.0) -> total should be 90.0
    assert res.total_score == 90.0
    assert res.passed is True

def test_update_stop_qr_finishes_early(test_controller, monkeypatch):
    test_controller._persist = MagicMock()
    test_controller.start_test("cand_002")
    
    # Send one maneuver
    mr1 = ManeuverResult(maneuver_name="straight_1", confidence=1.0, payload="test", bbox=None)
    test_controller.update(maneuver_result=mr1, frame_score=None)
    
    # Send "stop" QR
    mr_stop = ManeuverResult(maneuver_name="stop", confidence=1.0, payload="stop", bbox=None)
    res = test_controller.update(maneuver_result=mr_stop, frame_score=None)
    
    assert res is not None
    assert test_controller.state == TestState.FINISHED
    assert len(res.maneuvers) == 2 # 1 active + 1 closing on stop
