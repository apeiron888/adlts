"""
test_controller.py — Phase 5 TestController

State machine that drives the full test lifecycle.

States
------
IDLE      – waiting for POST /test/start
RUNNING   – actively scoring; transitions driven by QR events
FINISHED  – result computed, JSON written; awaiting next start

Integration
-----------
QueueConsumer._process() calls self.test_controller.update() every frame.
update() returns a TestResult (and transitions to FINISHED) when the last
maneuver closes; otherwise returns None.

main.py owns the lifecycle via POST /test/start, GET /test/status,
POST /test/abort.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from enum import Enum
from typing import Optional

from config import (
    MANEUVER_SEQUENCE,
    MANEUVER_WEIGHTS,
    PASS_THRESHOLD,
    RESULTS_DIR,
    FrameScore,
    ManeuverResult,
    ManeuverScore,
    MotionResult,
    TestResult,
    TrafficLightResult,
)

logger = logging.getLogger(__name__)


class TestState(str, Enum):
    IDLE     = "idle"
    RUNNING  = "running"
    FINISHED = "finished"


class TestController:
    """
    Drives a complete driving test lifecycle.

    Parameters
    ----------
    scoring_engine : ScoringEngine
        The live Phase 4 engine — TestController calls aggregate_maneuver()
        on it at each maneuver boundary.

    Usage
    -----
    controller.start_test("cand_001")        # activate from main.py endpoint
    result = controller.update(...)          # called every frame by QueueConsumer
    controller.abort()                       # cancel mid-test
    """

    def __init__(self, scoring_engine):
        self._se = scoring_engine

        self.state: TestState          = TestState.IDLE
        self.candidate_id: str         = ""
        self.started_at: float         = 0.0
        self._maneuver_index: int      = 0   # pointer into MANEUVER_SEQUENCE
        self._scores: list[ManeuverScore] = []

        os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Public lifecycle API ──────────────────────────────────────────────────

    def start_test(self, candidate_id: str) -> None:
        """
        Transition from IDLE to RUNNING.
        Resets all accumulators and the ScoringEngine buffer.
        Raises RuntimeError if called while already RUNNING.
        """
        if self.state == TestState.RUNNING:
            raise RuntimeError("A test is already in progress.")

        self.candidate_id    = candidate_id
        self.started_at      = time.monotonic()
        self._maneuver_index = 0
        self._scores         = []
        self._se._reset_buffer()   # clear any leftover frames from previous run
        self.state           = TestState.RUNNING

        logger.info(
            "TestController: test STARTED  candidate=%s  first_maneuver=%s",
            candidate_id, self.current_maneuver,
        )

    def abort(self) -> None:
        """Cancel the running test without saving a result."""
        if self.state == TestState.RUNNING:
            logger.warning(
                "TestController: test ABORTED at maneuver %d/%d (%s)",
                self._maneuver_index, len(MANEUVER_SEQUENCE), self.current_maneuver,
            )
        self.state = TestState.IDLE
        self._se._reset_buffer()

    # ── Per-frame update ─────────────────────────────────────────────────────

    def update(
        self,
        maneuver_result:      Optional[ManeuverResult],
        frame_score:          Optional[FrameScore],
        traffic_light_result: Optional[TrafficLightResult] = None,
        motion_result:        Optional[MotionResult]        = None,
        lane_result=None,
    ) -> Optional[TestResult]:
        """
        Called every frame by QueueConsumer._process().

        Returns
        -------
        None         – test still in progress (or IDLE / FINISHED).
        TestResult   – the single moment the test finishes.
        """
        if self.state != TestState.RUNNING:
            return None

        # A QR code signals the end of the current maneuver
        detected_name = (
            maneuver_result.maneuver_name
            if maneuver_result is not None
            else None
        )

        if detected_name is None:
            return None   # normal mid-maneuver frame

        # ── Maneuver boundary ─────────────────────────────────────────────────

        # Close the maneuver that just ended
        closing_name = self.current_maneuver  # may be None if this is the first QR
        if closing_name is not None:
            ms = self._se.aggregate_maneuver(closing_name)
            self._scores.append(ms)
            logger.info(
                "TestController: maneuver '%s' closed  final=%.1f  frames=%d",
                ms.name, ms.final_score, ms.frame_count,
            )

        # Check for "stop" sentinel — test over
        if detected_name == "stop" or self._all_done():
            return self._finish_test()

        # Advance to the next maneuver
        self._maneuver_index += 1
        if self._maneuver_index < len(MANEUVER_SEQUENCE):
            logger.info(
                "TestController: advancing to maneuver %d/%d — '%s'",
                self._maneuver_index + 1, len(MANEUVER_SEQUENCE), self.current_maneuver,
            )
        else:
            return self._finish_test()

        return None

    # ── Status helpers ───────────────────────────────────────────────────────

    @property
    def current_maneuver(self) -> Optional[str]:
        """Name of the maneuver currently being scored, or None."""
        if self._maneuver_index < len(MANEUVER_SEQUENCE):
            return MANEUVER_SEQUENCE[self._maneuver_index]
        return None

    @property
    def scores_so_far(self) -> list[ManeuverScore]:
        return list(self._scores)

    @property
    def progress(self) -> dict:
        """Snapshot for GET /test/status."""
        return {
            "state":            self.state.value,
            "candidate_id":     self.candidate_id,
            "current_maneuver": self.current_maneuver,
            "maneuver_index":   self._maneuver_index,
            "total_maneuvers":  len(MANEUVER_SEQUENCE),
            "scores_so_far":    [asdict(s) for s in self._scores],
            "frames_in_buffer": len(self._se._frame_scores),
        }

    # ── Private ──────────────────────────────────────────────────────────────

    def _all_done(self) -> bool:
        """True when we've already collected all 7 maneuver scores."""
        return len(self._scores) >= len(MANEUVER_SEQUENCE)

    def _finish_test(self) -> TestResult:
        """Compute final score, persist JSON, transition to FINISHED."""
        finished_at  = time.monotonic()
        total_score  = self._weighted_mean()
        passed       = total_score >= PASS_THRESHOLD

        result = TestResult(
            candidate_id = self.candidate_id,
            started_at   = self.started_at,
            finished_at  = finished_at,
            maneuvers    = list(self._scores),
            total_score  = round(total_score, 2),
            passed       = passed,
        )

        self._persist(result)
        self.state = TestState.FINISHED

        logger.info(
            "TestController: test FINISHED  candidate=%s  total=%.1f  passed=%s",
            self.candidate_id, total_score, passed,
        )
        return result

    def _weighted_mean(self) -> float:
        """Compute the weighted mean score across all completed maneuvers."""
        if not self._scores:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0
        for ms in self._scores:
            w = MANEUVER_WEIGHTS.get(ms.name, 1.0)
            weighted_sum += ms.final_score * w
            weight_total += w

        return weighted_sum / weight_total if weight_total > 0 else 0.0

    def _persist(self, result: TestResult) -> None:
        """
        Write the TestResult to a JSON file in RESULTS_DIR.

        Filename: <candidate_id>_<unix_ts_ms>.json
        The file is written atomically (temp + rename) to avoid partial reads.
        """
        ts_ms  = int(result.finished_at * 1000)
        fname  = f"{result.candidate_id}_{ts_ms}.json"
        fpath  = os.path.join(RESULTS_DIR, fname)
        tmp    = fpath + ".tmp"

        payload = {
            "candidate_id":  result.candidate_id,
            "started_at":    result.started_at,
            "finished_at":   result.finished_at,
            "total_score":   result.total_score,
            "passed":        result.passed,
            "pass_threshold": PASS_THRESHOLD,
            "maneuvers": [asdict(ms) for ms in result.maneuvers],
        }

        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, fpath)
            logger.info("TestController: result persisted → %s", fpath)
        except OSError as exc:
            logger.error("TestController: failed to persist result: %s", exc)
