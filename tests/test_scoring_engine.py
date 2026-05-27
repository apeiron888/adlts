import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from scoring_engine import ManeuverScore, ScoringEngine, VIOLATION_PENALTY
from config import TrafficLightResult, TrafficLightState, MotionResult

class DummyLaneDetector:
    def __init__(self):
        self.pixels_per_cm = None
        self.lane_width_px = None

@pytest.fixture
def calibrated_engine():
    ld = DummyLaneDetector()
    ld.pixels_per_cm = 2.0  # 2 pixels per cm
    return ScoringEngine(ld)

@pytest.fixture
def uncalibrated_engine():
    return ScoringEngine(DummyLaneDetector())

def test_uncalibrated_returns_none(uncalibrated_engine):
    assert uncalibrated_engine.score_frame((100, 240, 150, 100), (220, 240, 200, 100)) is None

def test_missing_line_returns_zero(calibrated_engine):
    fs = calibrated_engine.score_frame(None, (220, 240, 200, 100))
    assert fs.score == 0.0
    assert fs.error_cm == 999.0

def test_perfect_center(calibrated_engine):
    # car center is 160. So lane center = 160 is perfect.
    # left_line x1 = 130, right_line x1 = 190 -> center = 160
    left_line = (130, 240, 130, 150)
    right_line = (190, 240, 190, 150)
    fs = calibrated_engine.score_frame(left_line, right_line)
    assert fs.score == 100.0
    assert fs.error_cm == 0.0
    assert fs.error_px == 0.0

def test_max_drift(calibrated_engine):
    # Drift left by 15 cm. pixels_per_cm = 2.0 -> drift by 30 px.
    # car center 160, lane center should be 190.
    # left_line x1 = 160, right_line x1 = 220 -> center = 190.
    left_line = (160, 240, 160, 150)
    right_line = (220, 240, 220, 150)
    fs = calibrated_engine.score_frame(left_line, right_line)
    assert fs.score == 0.0
    assert fs.error_cm == 15.0
    
def test_half_drift(calibrated_engine):
    # Drift left by 7.5 cm = 15 px.
    # lane center = 175
    left_line = (145, 240, 145, 150)
    right_line = (205, 240, 205, 150)
    fs = calibrated_engine.score_frame(left_line, right_line)
    assert fs.score == 50.0
    assert fs.error_cm == 7.5

def test_traffic_event_streak(calibrated_engine):
    # RED + moving
    red_tl = TrafficLightResult(state=TrafficLightState.RED, confidence=1.0, bbox=None)
    moving = MotionResult(is_moving=True, pixel_change_ratio=0.05, changed_pixels=100, total_pixels=2000, roi=None)
    green_tl = TrafficLightResult(state=TrafficLightState.GREEN, confidence=1.0, bbox=None)
    stopped = MotionResult(is_moving=False, pixel_change_ratio=0.0, changed_pixels=0, total_pixels=2000, roi=None)

    # 4 frames -> no penalty
    for _ in range(4):
        assert not calibrated_engine.record_traffic_event(red_tl, moving)
    assert calibrated_engine._violations == 0

    # 5th frame -> triggers
    assert calibrated_engine.record_traffic_event(red_tl, moving)
    assert calibrated_engine._violations == 1
    assert calibrated_engine._penalty == VIOLATION_PENALTY

    # 6-9 frames -> no trigger
    for _ in range(4):
        assert not calibrated_engine.record_traffic_event(red_tl, moving)
    
    # 10th frame -> triggers again
    assert calibrated_engine.record_traffic_event(red_tl, moving)
    assert calibrated_engine._violations == 2
    assert calibrated_engine._penalty == VIOLATION_PENALTY * 2

    # green moving -> reset streak
    assert not calibrated_engine.record_traffic_event(green_tl, moving)
    assert calibrated_engine._red_moving_streak == 0

def test_aggregate_maneuver_small_sample(calibrated_engine):
    # < 10 frames -> simple mean
    calibrated_engine._frame_scores = [100.0, 90.0, 80.0]
    ms = calibrated_engine.aggregate_maneuver("straight_1")
    assert ms.raw_score == 90.0
    assert ms.final_score == 90.0
    assert ms.frame_count == 3
    # check reset
    assert not calibrated_engine._frame_scores

def test_aggregate_maneuver_trimmed_mean(calibrated_engine):
    # 10 frames: [0, 90, 90, 90, 90, 90, 90, 90, 90, 100]
    # Trim 10% (1 frame from each end). Sorted: 0, 90x8, 100.
    # Trims 0 and 100. Remaining: 8x 90. Mean = 90.
    calibrated_engine._frame_scores = [0.0, 100.0] + [90.0] * 8
    ms = calibrated_engine.aggregate_maneuver("straight_2")
    assert ms.raw_score == 90.0

def test_aggregate_maneuver_with_penalty(calibrated_engine):
    calibrated_engine._frame_scores = [100.0, 100.0, 100.0]
    calibrated_engine._penalty = 20.0
    ms = calibrated_engine.aggregate_maneuver("straight_3")
    assert ms.raw_score == 100.0
    assert ms.penalty == 20.0
    assert ms.final_score == 80.0

def test_parking_score(calibrated_engine):
    box_lines = {
        'left': (50, 0, 50, 100),
        'right': (250, 0, 250, 100)
    }
    # all corners inside
    car_corners_inside = [(100, 10), (200, 10), (100, 90), (200, 90)]
    score = calibrated_engine.score_parking(box_lines, car_corners_inside)
    # perfectly parallel -> alignment=50, inside=50 -> 100
    assert score == 100.0

    # one corner outside
    car_corners_outside = [(40, 10), (200, 10), (40, 90), (200, 90)]
    score2 = calibrated_engine.score_parking(box_lines, car_corners_outside)
    assert score2 == 50.0  # inside=0, alignment=50

    # poorly aligned box line (45 deg)
    box_lines_angled = {
        'left': (0, 0, 100, 100),  # dh=100, dw=100 -> 45deg
        'right': (250, 0, 250, 100)
    }
    score3 = calibrated_engine.score_parking(box_lines_angled, car_corners_inside)
    # alignment drops to 0 because angle is > 10
    assert score3 == 50.0  # inside=50, alignment=0
