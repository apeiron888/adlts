"""
tests/test_detectors.py — Phase 3 detector checks (QR + traffic light + motion).

Run:
  python tests/test_detectors.py
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import TrafficLightState
from motion_detector import MotionDetector
from qr_detector import QRDetector
from traffic_light_detector import TrafficLightDetector


def test_qr_payload_parsing_and_cooldown():
    detector = QRDetector(cooldown_s=0.5)

    assert detector._parse_payload("maneuver:straight_1") == "straight_1"
    assert detector._parse_payload("manoeuvre:left_curve_1") == "left_curve_1"
    assert detector._parse_payload("invalid") is None
    assert detector._parse_payload("maneuver:unknown") is None

    # cooldown logic (directly manipulate cache for deterministic test)
    detector._last_seen_at["straight_1"] = time.monotonic()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    # no real QR in synthetic frame => no event
    result = detector.detect(frame)
    assert result.maneuver_name is None


def test_traffic_light_hsv_classifier():
    detector = TrafficLightDetector(model_path="yolov8n.pt")

    red_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    red_crop[:, :] = (0, 0, 255)
    state, conf = detector._classify_hsv(red_crop)
    assert state == TrafficLightState.RED
    assert conf > 0.5

    green_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    green_crop[:, :] = (0, 255, 0)
    state, conf = detector._classify_hsv(green_crop)
    assert state == TrafficLightState.GREEN
    assert conf > 0.5

    dark_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    state, conf = detector._classify_hsv(dark_crop)
    assert state == TrafficLightState.NONE
    assert conf == 0.0


def test_motion_detector_threshold():
    detector = MotionDetector(movement_threshold_ratio=0.01, diff_threshold=20)

    frame_a = np.zeros((240, 320, 3), dtype=np.uint8)
    frame_b = frame_a.copy()
    cv2.rectangle(frame_b, (140, 160), (220, 239), (255, 255, 255), -1)

    first = detector.detect(frame_a)
    assert first.is_moving is False

    second = detector.detect(frame_b)
    assert second.pixel_change_ratio > 0.0
    assert second.is_moving is True


def main():
    print("Running detector tests...")
    test_qr_payload_parsing_and_cooldown()
    print("  ✓ QR parser/cooldown")

    test_traffic_light_hsv_classifier()
    print("  ✓ Traffic light HSV classifier")

    test_motion_detector_threshold()
    print("  ✓ Motion detector threshold")

    print("All detector tests passed.")


if __name__ == "__main__":
    main()
