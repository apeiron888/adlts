"""
qr_detector.py — Phase 3 QR maneuver detector.

Detects QR payloads like:
    maneuver:straight_1

Cooldown suppresses repeated triggers when the same QR stays visible
across many consecutive frames.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from config import MANEUVER_SEQUENCE, ManeuverResult

logger = logging.getLogger(__name__)


class QRDetector:
    """OpenCV QR code based maneuver detector."""

    def __init__(self, cooldown_s: float = 3.0):
        self.cooldown_s = cooldown_s
        self._qr = cv2.QRCodeDetector()
        self._last_seen_at: dict[str, float] = {}

    @staticmethod
    def _parse_payload(payload: str | None) -> Optional[str]:
        """
        Parse payload forms:
          - maneuver:<name>
          - manoeuvre:<name>
        Returns maneuver name if valid and known, else None.
        """
        if not payload:
            return None

        text = payload.strip().lower()
        if ":" not in text:
            return None

        prefix, value = text.split(":", 1)
        prefix = prefix.strip()
        value = value.strip()

        if prefix not in {"maneuver", "manoeuvre"}:
            return None

        # accept stop in addition to configured sequence
        allowed = set(MANEUVER_SEQUENCE) | {"stop"}
        if value not in allowed:
            return None

        return value

    @staticmethod
    def _points_to_bbox(points: np.ndarray | None) -> Optional[tuple[int, int, int, int]]:
        if points is None:
            return None

        pts = np.asarray(points).reshape(-1, 2)
        if pts.size == 0:
            return None

        x1 = int(np.min(pts[:, 0]))
        y1 = int(np.min(pts[:, 1]))
        x2 = int(np.max(pts[:, 0]))
        y2 = int(np.max(pts[:, 1]))
        return (x1, y1, x2, y2)

    def detect(self, frame: np.ndarray) -> ManeuverResult:
        """Detect and decode one QR maneuver payload from the frame."""
        payload, points, _ = self._qr.detectAndDecode(frame)
        maneuver_name = self._parse_payload(payload)

        # Log raw payload for debugging (even if invalid)
        if payload:
            logger.debug(f"QR raw payload: '{payload}' → parsed={maneuver_name}")

        if not maneuver_name:
            return ManeuverResult(
                maneuver_name=None,
                confidence=0.0,
                payload=None,
                bbox=None,
            )

        now = time.monotonic()
        last_seen = self._last_seen_at.get(maneuver_name, 0.0)
        if (now - last_seen) < self.cooldown_s:
            return ManeuverResult(
                maneuver_name=None,
                confidence=0.0,
                payload=payload,
                bbox=self._points_to_bbox(points),
            )

        self._last_seen_at[maneuver_name] = now
        return ManeuverResult(
            maneuver_name=maneuver_name,
            confidence=1.0,
            payload=payload,
            bbox=self._points_to_bbox(points),
        )
