"""
config.py — Single source of truth for all constants and data contracts.

ALL phases import from here.  No phase defines its own structures.

Phase 1  uses: TimestampedFrame, CAMERA_FPS, QUEUE_SAMPLE_FPS, QUEUE_MAX_SIZE,
               IMAGE_WIDTH, IMAGE_HEIGHT, MINIO_*, ESP32_STREAM_URL
Phase 2+ uses: LaneResult, LaneDetector params
Phase 3+ uses: ManeuverResult, TrafficLightResult, MotionResult,
               TrafficLightState, MANEUVER_SEQUENCE
Phase 4+ uses: FrameScore, LANE_WIDTH_CM
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
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

ESP32_STREAM_URL = os.getenv("ESP32_STREAM_URL", "http://192.168.1.10/stream")

# ─── Model paths (offline-friendly) ─────────────────────────────────────────

# Absolute path to avoid cwd issues
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
TRAFFIC_LIGHT_MODEL_PATH = os.getenv(
    "TRAFFIC_LIGHT_MODEL_PATH",
    os.path.join(_MODEL_DIR, "yolov8n.pt")
)

# ─── Debug output ─────────────────────────────────────────────────────────────

DEBUG_FRAMES_DIR = os.getenv("DEBUG_FRAMES_DIR", "debug_frames")

# ─── Lane / scoring ───────────────────────────────────────────────────────────

LANE_WIDTH_CM = 30.0   # Physical lane width — update after calibration run

# ─── Manoeuvre sequence ───────────────────────────────────────────────────────

MANEUVER_SEQUENCE = [
    "straight_1",
    "left_curve_1",
    "left_curve_2",
    "straight_2",
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


class TrafficLightState(str, Enum):
    """Traffic light classification state for the current frame."""
    RED = "red"
    GREEN = "green"
    NONE = "none"


@dataclass
class ManeuverResult:
    """
    Output of QRDetector.detect().
    """
    maneuver_name: Optional[str]   # e.g. "straight_1", or None when not found
    confidence: float              # QR detector has no score → 1.0 on valid decode
    payload: Optional[str]         # raw decoded payload, e.g. "maneuver:straight_1"
    bbox: Optional[tuple]          # (x1, y1, x2, y2) or None


@dataclass
class TrafficLightResult:
    """
    Output of TrafficLightDetector.detect().
    """
    state: TrafficLightState
    confidence: float
    bbox: Optional[tuple]          # (x1, y1, x2, y2) or None


@dataclass
class MotionResult:
    """
    Output of MotionDetector.detect().
    """
    is_moving: bool
    pixel_change_ratio: float
    changed_pixels: int
    total_pixels: int
    roi: Optional[tuple] = None    # (x1, y1, x2, y2) or None


@dataclass
class FrameScore:
    """
    Output of ScoringEngine.score_frame().
    Phase 4 fills this; earlier phases pass None.
    """
    score:    float   # 0–100
    error_cm: float   # lateral deviation in real-world cm
    error_px: float   # lateral deviation in pixels


@dataclass
class ManeuverScore:
    """
    Final score for one completed maneuver segment.
    Produced by ScoringEngine.aggregate_maneuver(); collected by Phase 5 TestController.
    """
    name:        str    # e.g. "straight_1"
    raw_score:   float  # trimmed mean of frame scores before penalties, 0–100
    penalty:     float  # total points deducted (traffic violations, etc.)
    final_score: float  # max(0, raw_score - penalty)
    frame_count: int    # number of frames scored during this maneuver
    violations:  int    # number of confirmed RED-light running violations


# ─── Phase 5 constants ────────────────────────────────────────────────────────

PASS_THRESHOLD = float(os.getenv("PASS_THRESHOLD", "60.0"))

RESULTS_DIR = os.getenv("RESULTS_DIR", "test_results")

# Relative weights per maneuver used when computing the weighted mean total score.
# Maneuvers that are harder / more safety-critical carry more weight.
MANEUVER_WEIGHTS: dict = {
    "straight_1":       1.0,
    "left_curve_1":     1.5,
    "straight_2":       1.0,
    "left_curve_2":     1.5,
    "straight_3":       1.0,
    "figure_8":         2.0,
    "parallel_parking": 2.0,
}


@dataclass
class TestResult:
    """
    Final output of one complete driving test.
    Produced by TestController._finish_test(); serialised to JSON by Phase 5.
    Later passed to the Go API for storage in the candidate record.
    """
    candidate_id:  str
    started_at:    float              # time.monotonic() at test start
    finished_at:   float              # time.monotonic() at test end
    maneuvers:     list               # List[ManeuverScore] in execution order
    total_score:   float              # weighted mean of ManeuverScore.final_score
    passed:        bool               # total_score >= PASS_THRESHOLD