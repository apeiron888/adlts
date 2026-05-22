"""
config.py — Single source of truth for all constants and data contracts.

ALL phases import from here.  No phase defines its own structures.

Phase 1  uses: TimestampedFrame, CAMERA_FPS, QUEUE_SAMPLE_FPS, QUEUE_MAX_SIZE,
               IMAGE_WIDTH, IMAGE_HEIGHT, MINIO_*, ESP32_STREAM_URL
Phase 2+ uses: LaneResult, LaneDetector params
Phase 3+ uses: SignResult, MANEUVER_SEQUENCE
Phase 4+ uses: FrameScore, LANE_WIDTH_CM
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# ─── Stream / queue tuning ────────────────────────────────────────────────────

CAMERA_FPS       = 30      # ESP32-CAM capture rate (frames/sec)
QUEUE_SAMPLE_FPS = 10      # Frames pushed to the processing queue per second
#                            Every (CAMERA_FPS // QUEUE_SAMPLE_FPS) = 3rd frame
QUEUE_MAX_SIZE   = 300     # Max queue depth ≈ 30 s of buffered frames at 10 fps

# ─── Camera resolution (must match ESP32 firmware) ───────────────────────────

IMAGE_WIDTH  = 320
IMAGE_HEIGHT = 240

# ─── MinIO ───────────────────────────────────────────────────────────────────

MINIO_ENDPOINT  = os.getenv("MINIO_ENDPOINT",  "localhost:9000")
MINIO_ACCESS    = os.getenv("MINIO_ACCESS",     "minioadmin")
MINIO_SECRET    = os.getenv("MINIO_SECRET",     "minioadmin")
MINIO_BUCKET    = "recordings"
MINIO_SECURE    = False    # set True with TLS in production

# ─── ESP32 stream ─────────────────────────────────────────────────────────────

ESP32_STREAM_URL = os.getenv("ESP32_STREAM_URL", "http://192.168.1.xxx/stream")

# ─── Debug output ─────────────────────────────────────────────────────────────

DEBUG_FRAMES_DIR = os.getenv("DEBUG_FRAMES_DIR", "debug_frames")

# ─── Lane / scoring ───────────────────────────────────────────────────────────

LANE_WIDTH_CM = 30.0   # Physical lane width — update after calibration run

# ─── Manoeuvre sequence ───────────────────────────────────────────────────────

MANEUVER_SEQUENCE = [
    "straight_1",
    "left_curve_1",
    "straight_2",
    "left_curve_2",
    "straight_3",
    "figure_8",
    "parallel_parking",
]

# ─────────────────────────────────────────────────────────────────────────────
# Data contracts — defined once, imported everywhere
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TimestampedFrame:
    """
    A single decoded frame plus a monotonic timestamp.
    Produced by StreamReceiver (Thread A), consumed by QueueConsumer (Thread B).
    """
    frame:        np.ndarray   # raw BGR image from OpenCV (copy — Thread A does not own it)
    timestamp_ms: float        # time.monotonic_ns() / 1e6 at the moment of capture


@dataclass
class LaneResult:
    """
    Output of LaneDetector.detect().
    Phase 2 fills this; Phase 1 QueueConsumer has a stub that passes None.
    """
    left_line:  Optional[tuple]   # (x1, y1, x2, y2) pixel coords, or None
    right_line: Optional[tuple]   # (x1, y1, x2, y2) pixel coords, or None
    centre_x:   Optional[float]   # lane centre x at the bottom of the frame, or None
    raw_frame:  np.ndarray        # original frame with optional debug overlay drawn on it


@dataclass
class SignResult:
    """
    Output of SignDetector.detect().
    Phase 3 fills this; Phase 1–2 QueueConsumer has a stub that passes None.
    """
    sign_id:    Optional[int]   # index into MANEUVER_SEQUENCE, -1 = stop sign, None = no sign
    confidence: float
    bbox:       Optional[tuple] # (x1, y1, x2, y2) or None


@dataclass
class FrameScore:
    """
    Output of ScoringEngine.score_frame().
    Phase 4 fills this; earlier phases pass None.
    """
    score:    float   # 0–100
    error_cm: float   # lateral deviation in real-world cm
    error_px: float   # lateral deviation in pixels