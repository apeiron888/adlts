"""
scoring_engine.py — Phase 4 Scoring Engine

Responsibilities
----------------
1. score_frame()           — lateral error → FrameScore (0–100) per frame.
2. record_traffic_event()  — buffer RED+moving observations frame by frame.
3. add_frame_score()       — accumulate into the active maneuver buffer.
4. aggregate_maneuver()    — trimmed mean + penalties → ManeuverScore; resets buffer.
5. score_parking()         — snapshot scorer for parallel parking (Phase 5 provides corners).

Design rules
------------
- Pure computation. No state machine. No QR / sign logic. No threading.
- Returns None for score_frame() when not yet calibrated — never emits garbage.
- Trimmed mean (10 % each end) is used instead of simple mean so that brief
  lane-detection glitches (0-score frames) do not unfairly drag down a good run.
- Traffic-light violations require VIOLATION_MIN_FRAMES consecutive RED+moving
  frames to be confirmed, avoiding single-frame noise.

Phase 5 (TestController) integration
-------------------------------------
  Each frame:
      fs = engine.score_frame(left_line, right_line)
      engine.record_traffic_event(tl_result, motion_result)
      if fs: engine.add_frame_score(fs)

  On QR-triggered maneuver boundary:
      ms = engine.aggregate_maneuver(maneuver_name)   # returns ManeuverScore, resets

  For parallel parking (car stopped in box):
      score = engine.score_parking(box_lines, car_corners)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from config import (
    IMAGE_WIDTH,
    LANE_WIDTH_CM,
    FrameScore,
    ManeuverScore,
    MotionResult,
    TrafficLightResult,
    TrafficLightState,
)

logger = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────

# Points deducted per confirmed RED-light running violation.
VIOLATION_PENALTY = 20.0

# Number of consecutive frames with RED+moving required to confirm a violation.
# At 10 fps → 5 frames = 0.5 s.  Prevents single-frame noise from firing.
VIOLATION_MIN_FRAMES = 5

# Fraction of frames trimmed from each end during maneuver aggregation.
# 0.10 = discard worst 10 % and best 10 % before computing the mean.
TRIM_RATIO = 0.10

# Minimum frame count before trimming is applied.
# Below this, a plain mean is used (trimming ≤3 items is meaningless).
TRIM_MIN_FRAMES = 10

# Maximum tolerable lateral error in cm (half the lane width).
MAX_ERROR_CM = LANE_WIDTH_CM / 2.0  # default 15 cm


# ─────────────────────────────────────────────────────────────────────────────

class ScoringEngine:
    """
    Computes per-frame lateral scores and aggregates them into maneuver scores.

    Parameters
    ----------
    lane_detector : LaneDetector
        Shares calibration attributes ``pixels_per_cm`` and ``lane_width_px``.
        ScoringEngine does NOT call lane_detector.detect() — it only reads
        calibration values already stored there.
    """

    def __init__(self, lane_detector):
        self._ld = lane_detector

        # ── Active maneuver accumulator ───────────────────────────────────────
        self._frame_scores: list[float] = []
        self._penalty: float = 0.0
        self._violations: int = 0

        # ── Traffic-light violation streak counter ────────────────────────────
        # Incremented each frame when RED+moving; resets on any non-violation frame.
        # Fires (adds penalty) exactly when streak reaches VIOLATION_MIN_FRAMES.
        self._red_moving_streak: int = 0

    # ── Calibration guard ─────────────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        """True once LaneDetector.calibrate() has been called successfully."""
        return self._ld.pixels_per_cm is not None

    @property
    def pixels_per_cm(self) -> Optional[float]:
        return self._ld.pixels_per_cm

    # ── Per-frame scoring ─────────────────────────────────────────────────────

    def score_frame(
        self,
        left_line: Optional[tuple],
        right_line: Optional[tuple],
    ) -> Optional[FrameScore]:
        """
        Compute lateral deviation score for one frame.

        Algorithm
        ---------
        1. Measure lane centre at the bottom of the frame (x1 of each line).
        2. car_centre = IMAGE_WIDTH / 2  (camera is mounted at car centre line).
        3. error_px = |car_centre - lane_centre|
        4. error_cm = error_px / pixels_per_cm
        5. score   = 100 × max(0,  1 − error_cm / MAX_ERROR_CM)

        Returns
        -------
        None            — if not yet calibrated (never emit garbage).
        FrameScore      — with score=0, error_cm=999 if both lines missing.
        FrameScore(0–100) — normal result.
        """
        if not self.is_calibrated:
            logger.debug("score_frame: not calibrated — returning None")
            return None

        if left_line is None or right_line is None:
            logger.debug("score_frame: one or both lines missing — score=0")
            return FrameScore(score=0.0, error_cm=999.0, error_px=999.0)

        # x1 is the bottom intercept of each extrapolated line (closest to car)
        lane_centre_x = (left_line[0] + right_line[0]) / 2.0
        car_centre_x  = IMAGE_WIDTH / 2.0

        error_px  = abs(car_centre_x - lane_centre_x)
        error_cm  = error_px / self.pixels_per_cm          # type: ignore[operator]
        score     = 100.0 * max(0.0, 1.0 - error_cm / MAX_ERROR_CM)

        logger.debug(
            "score_frame: error_px=%.1f  error_cm=%.2f  score=%.1f",
            error_px, error_cm, score,
        )
        return FrameScore(score=score, error_cm=error_cm, error_px=error_px)

    # ── Traffic-light violation tracker ──────────────────────────────────────

    def record_traffic_event(
        self,
        tl_result: Optional[TrafficLightResult],
        motion_result: Optional[MotionResult],
    ) -> bool:
        """
        Call once per frame during an active maneuver *after* score_frame().

        Tracks consecutive RED+moving frames.  When the streak reaches
        VIOLATION_MIN_FRAMES, a violation is confirmed and VIOLATION_PENALTY
        points are added to the current maneuver's penalty accumulator.

        Returns True the single moment a violation is confirmed (useful for
        logging / dashboard alerts); False in all other cases.

        Edge cases
        ----------
        - Either argument is None → treat as no event, reset streak.
        - RED but car stops (is_moving=False) → streak resets; no penalty.
        - GREEN or NONE while moving → streak resets; not a violation.
        - Streak reaches threshold multiple times (e.g. 10 frames) → only fires
          on exactly frame 5, then again on exactly frame 10, etc.
          (streak is NOT reset after confirming — subsequent violations are
          possible if the car keeps running the light after the first penalty.)
        """
        if tl_result is None or motion_result is None:
            self._red_moving_streak = 0
            return False

        # Do not track violations before calibration — no test is underway yet.
        if not self.is_calibrated:
            return False

        if (tl_result.state == TrafficLightState.RED
                and motion_result.is_moving):
            self._red_moving_streak += 1
            logger.debug(
                "record_traffic_event: RED+moving streak=%d", self._red_moving_streak
            )

            # Fire exactly when reaching the threshold (and every multiple after)
            if self._red_moving_streak % VIOLATION_MIN_FRAMES == 0:
                self._penalty += VIOLATION_PENALTY
                self._violations += 1
                logger.warning(
                    "Traffic violation confirmed! streak=%d violations=%d penalty=%.0f",
                    self._red_moving_streak, self._violations, self._penalty,
                )
                return True
        else:
            if self._red_moving_streak > 0:
                logger.debug(
                    "record_traffic_event: streak reset (was %d)", self._red_moving_streak
                )
            self._red_moving_streak = 0

        return False

    # ── Maneuver score accumulation ───────────────────────────────────────────

    def add_frame_score(self, fs: FrameScore) -> None:
        """
        Append a FrameScore to the active maneuver buffer.
        Call this for every frame where score_frame() returns a real score.
        """
        self._frame_scores.append(fs.score)

    def aggregate_maneuver(self, name: str) -> ManeuverScore:
        """
        Finalise the current maneuver and return its aggregated score.

        Algorithm
        ---------
        - < TRIM_MIN_FRAMES frames: plain mean (too few for trimming to be meaningful).
        - ≥ TRIM_MIN_FRAMES frames: trimmed mean — discard bottom TRIM_RATIO %
          and top TRIM_RATIO % before computing the mean.  This makes the score
          robust to brief lane-detection glitches (single-frame 0 s or 100 s).
        - Penalties (traffic violations) are subtracted from the trimmed mean.
        - Final score is clamped to [0, 100].

        Resets the internal buffer and penalty counters after computing.

        Called by Phase 5 TestController when a QR code signals the next maneuver.
        """
        scores = list(self._frame_scores)

        if not scores:
            raw = 0.0
            logger.warning("aggregate_maneuver: no frames recorded for '%s'", name)
        elif len(scores) < TRIM_MIN_FRAMES:
            raw = float(np.mean(scores))
            logger.info(
                "aggregate_maneuver '%s': plain mean (n=%d) raw=%.1f",
                name, len(scores), raw,
            )
        else:
            trim = max(1, int(len(scores) * TRIM_RATIO))
            trimmed = sorted(scores)[trim: len(scores) - trim]
            raw = float(np.mean(trimmed)) if trimmed else float(np.mean(scores))
            logger.info(
                "aggregate_maneuver '%s': trimmed mean (n=%d trim=%d) raw=%.1f",
                name, len(scores), trim, raw,
            )

        final = float(np.clip(raw - self._penalty, 0.0, 100.0))

        result = ManeuverScore(
            name        = name,
            raw_score   = round(raw, 2),
            penalty     = round(self._penalty, 2),
            final_score = round(final, 2),
            frame_count = len(scores),
            violations  = self._violations,
        )
        logger.info(
            "ManeuverScore: %s  raw=%.1f  penalty=%.0f  final=%.1f  frames=%d  violations=%d",
            name, raw, self._penalty, final, len(scores), self._violations,
        )

        self._reset_buffer()
        return result

    def _reset_buffer(self) -> None:
        """Reset per-maneuver accumulators.  Does NOT reset streak (mid-maneuver state)."""
        self._frame_scores = []
        self._penalty      = 0.0
        self._violations   = 0
        self._red_moving_streak = 0   # also reset streak between maneuvers

    # ── Parallel parking snapshot scorer ─────────────────────────────────────

    def score_parking(
        self,
        box_lines: dict,
        car_corners: list,
    ) -> float:
        """
        One-shot score for parallel parking (NOT frame-accumulated).

        Called once by Phase 5 TestController after the car has stopped inside
        the parking box.  Phase 5 is responsible for detecting the stop and
        providing the corner coordinates via computer vision.

        Parameters
        ----------
        box_lines : dict with keys 'left', 'right', 'back'
            Each value is a (x1, y1, x2, y2) line tuple detected in the frame.
        car_corners : list of 4 (x, y) pixel coordinates
            Approximate corners of the car chassis (from Phase 5 detection).

        Returns
        -------
        float : 0–100
            50 pts if all corners are inside the box (binary).
            50 pts scaled by alignment quality (angle vs. box wall).

        Edge cases
        ----------
        - Missing box_lines key → that sub-score returns 0.
        - car_corners empty or malformed → inside_score = 0.
        - Perfect alignment (0°) → alignment_score = 50.
        - 10° or more misalignment → alignment_score = 0.
        """
        inside_score    = self._check_inside(box_lines, car_corners) * 50.0
        alignment_score = self._check_alignment(box_lines) * 50.0
        total = inside_score + alignment_score

        logger.info(
            "score_parking: inside=%.0f/50  alignment=%.1f/50  total=%.1f/100",
            inside_score, alignment_score, total,
        )
        return total

    def _check_inside(self, box_lines: dict, car_corners: list) -> float:
        """
        Returns 1.0 if ALL car corners are strictly between box left and right x.
        Returns 0.0 on any corner outside or on any exception.
        """
        try:
            left_x  = box_lines["left"][0]
            right_x = box_lines["right"][0]
            if left_x >= right_x:
                logger.warning("_check_inside: left_x >= right_x (%d >= %d)", left_x, right_x)
                return 0.0
            return 1.0 if all(left_x < cx < right_x for cx, _cy in car_corners) else 0.0
        except (KeyError, TypeError, IndexError) as exc:
            logger.error("_check_inside error: %s", exc)
            return 0.0

    def _check_alignment(self, box_lines: dict) -> float:
        """
        Returns a 0–1 quality factor based on how parallel the left box line is
        to the vertical axis.  Perfect vertical = 1.0; 10° or more = 0.0.
        """
        try:
            x1, y1, x2, y2 = box_lines["left"]
            # arctan2(Δx, Δy) gives angle from vertical
            angle_deg = abs(np.degrees(np.arctan2(x2 - x1, y2 - y1)))
            return float(max(0.0, 1.0 - angle_deg / 10.0))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            logger.error("_check_alignment error: %s", exc)
            return 0.0
