"""
lane_detector.py — Phase 2 Lane Detector

Classical OpenCV pipeline for white-line lane detection:
  1. Grayscale + Gaussian blur
  2. Canny edge detection
  3. Trapezoid ROI mask
  4. Hough line detection
  5. Separate by slope into left/right lines
  6. Average and extrapolate each side
  7. Fallback estimation when one side is missing
  8. Compute lane centre at bottom of frame
  9. Draw overlay with green lines + red center marker

Calibration captures the pixel width of the lane at the bottom,
allowing Phase 4 ScoringEngine to convert pixel error to cm.
"""

import cv2
import numpy as np
from config import LaneResult, IMAGE_WIDTH, IMAGE_HEIGHT


class LaneDetector:
    """
    Classical OpenCV lane detection for white lines on a dark surface.
    Returns a LaneResult with left/right lines, centre_x, and an overlay frame.
    """

    # Tunable parameters (can be adjusted via config.py later)
    CANNY_LOW = 50
    CANNY_HIGH = 150
    HOUGH_RHO = 1
    HOUGH_THETA = np.pi / 180
    HOUGH_THRESHOLD = 20
    HOUGH_MIN_LEN = 20
    HOUGH_MAX_GAP = 10

    # Calibration values (set by calibrate())
    lane_width_px: float | None = None
    pixels_per_cm: float | None = None

    def calibrate(self, left_line: tuple, right_line: tuple, lane_width_cm: float):
        """
        Call once when the car is centred on a straight lane.
        Stores the pixel width at the bottom of the image and computes pixels per cm.

        Args:
            left_line: (x1, y1, x2, y2) of detected left lane line
            right_line: (x1, y1, x2, y2) of detected right lane line
            lane_width_cm: Physical lane width in centimetres (e.g., 30.0)
        """
        lx = left_line[0]   # x1 of left line (bottom point)
        rx = right_line[0]  # x1 of right line (bottom point)
        self.lane_width_px = abs(rx - lx)
        self.pixels_per_cm = self.lane_width_px / lane_width_cm

    def detect(self, frame: np.ndarray) -> LaneResult:
        """
        Detect lanes in a single frame.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            LaneResult with left_line, right_line, centre_x, raw_frame (overlay)
        """
        h, w = frame.shape[:2]

        # 1. Preprocessing: gray → blur → Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.CANNY_LOW, self.CANNY_HIGH)

        # 2. Region-of-interest mask (trapezoid: bottom-wide, top-narrow)
        roi_vertices = np.array([[
            (0, h),
            (int(w * 0.1), int(h * 0.55)),
            (int(w * 0.9), int(h * 0.55)),
            (w, h),
        ]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked = cv2.bitwise_and(edges, mask)

        # 3. Hough line detection
        lines = cv2.HoughLinesP(
            masked,
            self.HOUGH_RHO, self.HOUGH_THETA, self.HOUGH_THRESHOLD,
            minLineLength=self.HOUGH_MIN_LEN,
            maxLineGap=self.HOUGH_MAX_GAP,
        )

        # 4. Separate lines into left and right by slope
        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.3:          # negative slope → left lane
                    left_lines.append(line[0])
                elif slope > 0.3:         # positive slope → right lane
                    right_lines.append(line[0])

        # 5. Average lines to get one left and one right line
        left_line  = self._average_lines(left_lines,  h) if left_lines  else None
        right_line = self._average_lines(right_lines, h) if right_lines else None

        # 6. Fallback: if one side is missing, estimate using calibration
        if left_line is None and right_line is not None and self.lane_width_px:
            rx1 = right_line[0]
            left_line = (
                int(rx1 - self.lane_width_px), h,
                int(rx1 - self.lane_width_px * 0.5), int(h * 0.6)
            )
        elif right_line is None and left_line is not None and self.lane_width_px:
            lx1 = left_line[0]
            right_line = (
                int(lx1 + self.lane_width_px), h,
                int(lx1 + self.lane_width_px * 0.5), int(h * 0.6)
            )

        # 7. Lane centre at bottom of frame
        centre_x = None
        if left_line and right_line:
            centre_x = (left_line[0] + right_line[0]) / 2.0

        # 8. Overlay for debugging
        overlay = frame.copy()
        if left_line:
            cv2.line(overlay, left_line[:2], left_line[2:], (0, 255, 0), 2)
        if right_line:
            cv2.line(overlay, right_line[:2], right_line[2:], (0, 255, 0), 2)
        if centre_x is not None:
            cv2.circle(overlay, (int(centre_x), h - 10), 5, (0, 0, 255), -1)

        return LaneResult(
            left_line=left_line,
            right_line=right_line,
            centre_x=centre_x,
            raw_frame=overlay,
        )

    @staticmethod
    def _average_lines(lines: list, img_height: int):
        """
        Average a list of line segments into one extrapolated line that spans
        from the bottom of the image to 60% of the height.

        Args:
            lines: List of (x1, y1, x2, y2) tuples
            img_height: Height of the image in pixels

        Returns:
            (x1, y1, x2, y2) tuple representing the averaged line, or None
        """
        if not lines:
            return None

        slopes, intercepts = [], []
        for x1, y1, x2, y2 in lines:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            slopes.append(slope)
            intercepts.append(intercept)

        if not slopes:
            return None

        avg_slope = float(np.mean(slopes))
        avg_intercept = float(np.mean(intercepts))

        y1 = img_height
        y2 = int(img_height * 0.6)

        # Avoid division by zero (should not happen because we filter vertical lines)
        if abs(avg_slope) < 1e-6:
            avg_slope = 1e-6 if avg_slope >= 0 else -1e-6

        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)

        return (x1, y1, x2, y2)
