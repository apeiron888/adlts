"""
motion_detector.py — frame-difference movement detector.

Used with traffic light logic:
  RED + moving   => violation
  RED + not move => good stop
  GREEN + moving => good go
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import MotionResult


class MotionDetector:
    """Simple pixel-change based movement detector using frame differencing."""

    def __init__(
        self,
        movement_threshold_ratio: float = 0.01,
        diff_threshold: int = 25,
        blur_kernel: int = 5,
        roi: Optional[tuple[int, int, int, int]] = None,
    ):
        self.movement_threshold_ratio = movement_threshold_ratio
        self.diff_threshold = diff_threshold
        self.blur_kernel = blur_kernel
        self.roi = roi
        self._prev_gray: np.ndarray | None = None

    def _apply_roi(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            x1 = max(0, min(w - 1, x1))
            x2 = max(x1 + 1, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(y1 + 1, min(h, y2))
            return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

        # default ROI = lower half of image where car motion is most visible
        y1 = h // 2
        return frame[y1:h, 0:w], (0, y1, w, h)

    def detect(self, frame: np.ndarray) -> MotionResult:
        roi_frame, roi_bbox = self._apply_roi(frame)

        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            total_pixels = gray.size
            return MotionResult(
                is_moving=False,
                pixel_change_ratio=0.0,
                changed_pixels=0,
                total_pixels=total_pixels,
                roi=roi_bbox,
            )

        diff = cv2.absdiff(self._prev_gray, gray)
        _, binary = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        changed_pixels = int(np.count_nonzero(binary))
        total_pixels = int(binary.size)
        ratio = (changed_pixels / total_pixels) if total_pixels else 0.0
        is_moving = ratio >= self.movement_threshold_ratio

        self._prev_gray = gray

        return MotionResult(
            is_moving=is_moving,
            pixel_change_ratio=ratio,
            changed_pixels=changed_pixels,
            total_pixels=total_pixels,
            roi=roi_bbox,
        )
