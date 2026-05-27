"""
mock_detectors.py — mock detectors for end-to-end pipeline testing without hardware.
"""

from __future__ import annotations

from itertools import cycle

from config import ManeuverResult, MotionResult, TrafficLightResult, TrafficLightState


class MockQRDetector:
    """Returns a maneuver event every N frames from a cyclic sequence."""

    def __init__(self, every_n_frames: int = 30):
        self.every_n_frames = every_n_frames
        self._index = 0
        self._count = 0
        self._sequence = cycle([
            "straight_1",
            "left_curve_1",
            "straight_2",
            "left_curve_2",
            "straight_3",
            "figure_8",
            "parallel_parking",
            "stop",
        ])

    def detect(self, frame):
        self._count += 1
        if self._count % self.every_n_frames != 0:
            return ManeuverResult(maneuver_name=None, confidence=0.0, payload=None, bbox=None)

        name = next(self._sequence)
        return ManeuverResult(
            maneuver_name=name,
            confidence=1.0,
            payload=f"maneuver:{name}",
            bbox=(10, 10, 60, 60),
        )


class MockTrafficLightDetector:
    """
    Simulates a realistic traffic light cycle.

    Cycle (frames): GREEN×60 → NONE×5 → RED×30 → NONE×5 → repeat
    The car is expected to stop during RED.  MockMotionDetector is coordinated
    with this so that RED+moving never fires by default.
    """

    CYCLE = [
        (TrafficLightState.GREEN, 60),
        (TrafficLightState.NONE,   5),
        (TrafficLightState.RED,   30),
        (TrafficLightState.NONE,   5),
    ]

    def __init__(self):
        self._frame = 0
        self._flat = []
        for state, n in self.CYCLE:
            self._flat.extend([state] * n)

    def detect(self, frame):
        state = self._flat[self._frame % len(self._flat)]
        self._frame += 1
        return TrafficLightResult(
            state=state,
            confidence=0.9 if state != TrafficLightState.NONE else 0.0,
            bbox=(200, 20, 240, 100) if state != TrafficLightState.NONE else None,
        )


class MockMotionDetector:
    """
    Mirrors MockTrafficLightDetector: car moves during GREEN, stops during RED.

    Same frame indices as MockTrafficLightDetector so is_moving is always False
    when state==RED → no traffic violations fire in normal mock operation.
    """

    CYCLE = [
        (True,  60),   # GREEN phase — moving
        (False,  5),   # transition
        (False, 30),   # RED phase — stopped
        (False,  5),   # transition
    ]

    def __init__(self):
        self._frame = 0
        self._flat = []
        for moving, n in self.CYCLE:
            self._flat.extend([moving] * n)

    def detect(self, frame):
        moving = self._flat[self._frame % len(self._flat)]
        self._frame += 1
        ratio = 0.03 if moving else 0.002
        return MotionResult(
            is_moving=moving,
            pixel_change_ratio=ratio,
            changed_pixels=int(1000 * ratio),
            total_pixels=1000,
            roi=(0, 120, 320, 240),
        )

