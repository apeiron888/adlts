"""
traffic_light_detector.py — YOLO + HSV traffic light detector.

1) YOLO (COCO class 9) localizes traffic light bbox.
2) HSV thresholds classify color inside selected bbox as RED/GREEN/NONE.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from config import TrafficLightResult, TrafficLightState

logger = logging.getLogger(__name__)


class TrafficLightDetector:
    """Detect traffic-light location with YOLO and classify color via HSV."""

    COCO_TRAFFIC_LIGHT_CLASS_ID = 9

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        min_color_ratio: float = 0.08,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_color_ratio = min_color_ratio
        self._model = None

        try:
            import os
            from ultralytics import YOLO  # lazy import to keep mock/dev mode lightweight

            # Check if local file exists before trying to load
            if not os.path.exists(model_path):
                logger.error(
                    "TrafficLightDetector: Model file not found at %s. "
                    "Please download from: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
                    model_path,
                )
                return

            self._model = YOLO(model_path)
            logger.info("TrafficLightDetector loaded model from %s", model_path)
        except Exception as exc:
            logger.error(
                "TrafficLightDetector: YOLO initialization failed (%s). Returning NONE state.",
                exc,
            )

    @staticmethod
    def _clip_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))
        return x1, y1, x2, y2

    def _classify_hsv(self, crop: np.ndarray) -> tuple[TrafficLightState, float]:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Red wraps around hue: [0..10] U [170..180]
        lower_red_1 = np.array([0, 80, 80], dtype=np.uint8)
        upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([170, 80, 80], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

        # Green
        lower_green = np.array([40, 80, 80], dtype=np.uint8)
        upper_green = np.array([80, 255, 255], dtype=np.uint8)

        red_mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(hsv, lower_red_2, upper_red_2)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        total = float(crop.shape[0] * crop.shape[1])
        if total <= 0:
            return TrafficLightState.NONE, 0.0

        red_ratio = float(np.count_nonzero(red_mask)) / total
        green_ratio = float(np.count_nonzero(green_mask)) / total

        if red_ratio >= self.min_color_ratio and red_ratio >= green_ratio:
            return TrafficLightState.RED, red_ratio
        if green_ratio >= self.min_color_ratio and green_ratio > red_ratio:
            return TrafficLightState.GREEN, green_ratio
        return TrafficLightState.NONE, max(red_ratio, green_ratio)

    def detect(self, frame: np.ndarray) -> TrafficLightResult:
        if self._model is None:
            return TrafficLightResult(
                state=TrafficLightState.NONE,
                confidence=0.0,
                bbox=None,
            )

        try:
            results = self._model(frame, verbose=False, conf=self.confidence_threshold)
        except Exception as exc:
            logger.error("TrafficLightDetector inference error: %s", exc)
            return TrafficLightResult(state=TrafficLightState.NONE, confidence=0.0, bbox=None)

        best = None  # (conf, (x1,y1,x2,y2))

        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls.item())
                if cls_id != self.COCO_TRAFFIC_LIGHT_CLASS_ID:
                    continue

                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                if best is None or conf > best[0]:
                    best = (conf, bbox)

        if best is None:
            return TrafficLightResult(
                state=TrafficLightState.NONE,
                confidence=0.0,
                bbox=None,
            )

        det_conf, bbox = best
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._clip_bbox(*bbox, w=w, h=h)
        crop = frame[y1:y2, x1:x2]

        state, color_conf = self._classify_hsv(crop)
        combined_conf = min(1.0, 0.5 * det_conf + 0.5 * color_conf)

        # DEBUG: Log color ratios to diagnose HSV threshold issues
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_red_1 = np.array([0, 80, 80], dtype=np.uint8)
        upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([170, 80, 80], dtype=np.uint8)
        upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
        lower_green = np.array([40, 80, 80], dtype=np.uint8)
        upper_green = np.array([80, 255, 255], dtype=np.uint8)
        red_mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(hsv, lower_red_2, upper_red_2)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        total = float(crop.shape[0] * crop.shape[1])
        red_ratio = float(np.count_nonzero(red_mask)) / total if total > 0 else 0.0
        green_ratio = float(np.count_nonzero(green_mask)) / total if total > 0 else 0.0
        logger.debug(
            "TrafficLightDetector: YOLO conf=%.2f, RED_ratio=%.3f GREEN_ratio=%.3f => %s (color_conf=%.3f)",
            det_conf, red_ratio, green_ratio, state.name, color_conf,
        )

        return TrafficLightResult(
            state=state,
            confidence=combined_conf,
            bbox=(x1, y1, x2, y2),
        )
