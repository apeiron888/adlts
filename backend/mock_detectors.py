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
    """Cycles traffic light states every N frames."""

    def __init__(self, every_n_frames: int = 45):
        self.every_n_frames = every_n_frames
        self._count = 0
        self._states = cycle([
            TrafficLightState.RED,
            TrafficLightState.NONE,
            TrafficLightState.GREEN,
            TrafficLightState.NONE,
        ])
        self._current = TrafficLightState.NONE

    def detect(self, frame):
        self._count += 1
        if self._count % self.every_n_frames == 0:
            self._current = next(self._states)

        return TrafficLightResult(
            state=self._current,
            confidence=0.9 if self._current != TrafficLightState.NONE else 0.0,
            bbox=(200, 20, 240, 100) if self._current != TrafficLightState.NONE else None,
        )


class MockMotionDetector:
    """Alternates motion state every N frames."""

    def __init__(self, every_n_frames: int = 30):
        self.every_n_frames = every_n_frames
        self._count = 0
        self._moving = False

    def detect(self, frame):
        self._count += 1
        if self._count % self.every_n_frames == 0:
            self._moving = not self._moving

        ratio = 0.03 if self._moving else 0.002
        return MotionResult(
            is_moving=self._moving,
            pixel_change_ratio=ratio,
            changed_pixels=int(1000 * ratio),
            total_pixels=1000,
            roi=(0, 120, 320, 240),
        )
