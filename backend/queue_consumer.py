"""
queue_consumer.py — Thread B: pops frames from FrameQueue and processes them.

Phase 1  (ACTIVE):
  • Saves every popped frame to debug_frames/<timestamp_ms>.jpg so you can
    visually confirm the queue is working.
  • Overlays timestamp and queue depth on the saved image.
  • Logs one line per frame to the console.

Phase 2+ (STUB — wired but inactive until the component is passed in):
  • LaneDetector.detect()     → LaneResult
    • QRDetector.detect()       → ManeuverResult
    • TrafficLightDetector.detect() → TrafficLightResult
    • MotionDetector.detect()   → MotionResult
  • ScoringEngine.score_frame()         → FrameScore        (Phase 4 ACTIVE)
  • ScoringEngine.record_traffic_event() → violation tracking (Phase 4 ACTIVE)
  • ScoringEngine.add_frame_score()      → maneuver buffer   (Phase 4 ACTIVE)
  • TestController.update()              → stub (Phase 5)
  • Dashboard.emit_update()              → stub (Phase 6)

To activate Phase 2, simply pass lane_detector=LaneDetector() to the
constructor.  No other change needed.  Same pattern for each later phase.

OMITTED (future phases):
  - Result persistence to MinIO / database
    - Overlay rendering pipeline (done in LaneDetector / QR/TrafficLight detectors)
  - WebSocket push of per-frame results (Phase 6)
"""

import logging
import os
import threading
import time

import cv2
import numpy as np

from config import DEBUG_FRAMES_DIR, TimestampedFrame

logger = logging.getLogger(__name__)


class QueueConsumer(threading.Thread):
    """
    Daemon thread — pops one frame at a time from frame_queue and runs the
    processing pipeline on it.

    Constructor arguments that are None are treated as "not yet available"
    (i.e. that phase has not been implemented yet).  Passing a real instance
    activates the corresponding pipeline stage.

    Parameters
    ----------
    frame_queue     : FrameQueue — shared with StreamReceiver (Thread A).
    lane_detector   : LaneDetector instance, or None (Phase 2).
    qr_detector     : QRDetector instance, or None (Phase 3).
    traffic_light_detector : TrafficLightDetector instance, or None (Phase 3).
    motion_detector : MotionDetector instance, or None (Phase 3).
    scoring_engine  : ScoringEngine instance, or None (Phase 4).
    test_controller : TestController instance, or None (Phase 5).
    dashboard       : Dashboard / Socket.IO emitter, or None (Phase 6).
    save_debug      : If True (default), save every popped frame to disk.
                      Set False in production once the pipeline is trusted.
    """

    def __init__(
        self,
        frame_queue,
        lane_detector=None,
        qr_detector=None,
        traffic_light_detector=None,
        motion_detector=None,
        scoring_engine=None,
        test_controller=None,
        dashboard=None,
        save_debug: bool = True,
        dashboard_emit_every_n: int = 3,
    ):
        super().__init__(daemon=True, name="QueueConsumer")
        self.frame_queue     = frame_queue
        self.lane_detector   = lane_detector    # None → Phase 2 not active
        self.qr_detector     = qr_detector      # None → Phase 3 not active
        self.traffic_light_detector = traffic_light_detector  # None → Phase 3 not active
        self.motion_detector = motion_detector  # None → Phase 3 not active
        self.scoring_engine  = scoring_engine   # None → Phase 4 not active
        self.test_controller = test_controller  # None → Phase 5 not active
        self.dashboard       = dashboard        # None → Phase 6 not active
        self.save_debug      = save_debug
        self.dashboard_emit_every_n = dashboard_emit_every_n  # Emit to dashboard every Nth frame

        self._stop_event = threading.Event()

        # Stats
        self.frames_processed = 0
        self._last_log_time   = time.monotonic()

        if self.save_debug:
            os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)
            logger.info(
                "QueueConsumer: debug frame saving enabled → %s/",
                os.path.abspath(DEBUG_FRAMES_DIR),
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()

    # ── Thread body ───────────────────────────────────────────────────────────

    def run(self):
        logger.info("QueueConsumer started.")
        while not self._stop_event.is_set():
            # Blocking pop — sleeps up to 0.5 s if queue is empty
            tsf: TimestampedFrame | None = self.frame_queue.pop(timeout=0.5)
            if tsf is None:
                continue   # timeout — check stop_event and try again

            self._process(tsf)
            self.frames_processed += 1

            # Log throughput every 5 s
            now = time.monotonic()
            if now - self._last_log_time >= 5.0:
                logger.info(
                    "QueueConsumer — processed=%d  queue_depth=%d  dropped=%d",
                    self.frames_processed,
                    self.frame_queue.size(),
                    self.frame_queue.dropped,
                )
                self._last_log_time = now

        # Drain remaining frames after stop signal
        logger.info(
            "QueueConsumer stop requested — draining %d buffered frames ...",
            self.frame_queue.size(),
        )
        drained = 0
        while True:
            tsf = self.frame_queue.pop(timeout=0.1)
            if tsf is None:
                break
            self._process(tsf)
            drained += 1
        logger.info("QueueConsumer drained %d frames. Exiting.", drained)

    # ── Processing pipeline ───────────────────────────────────────────────────

    def _process(self, tsf: TimestampedFrame):
        """
        Single-frame processing pipeline.
        Each stage is guarded by `if self.<component>:` so the pipeline
        gracefully degrades when future-phase components are not yet plugged in.
        """

        # ── Phase 2: Lane detection ──────────────────────────────────────────
        lane_result = None
        if self.lane_detector:
            try:
                lane_result = self.lane_detector.detect(tsf.frame)
            except Exception as exc:
                logger.error("LaneDetector error: %s", exc)

        # ── Phase 1+2+4: Save frame to disk ──────────────────────────────────
        # In Phase 2, we save the lane overlay if available; else save raw frame.
        # This lets us visually confirm the lane detection is working.
        if self.save_debug:
            frame_to_save = lane_result.raw_frame if lane_result else tsf.frame
            self._debug_save(frame_to_save, tsf.timestamp_ms, getattr(self, "_latest_score", None))

        # ── Phase 3: QR maneuver detection ───────────────────────────────────
        maneuver_result = None
        if self.qr_detector:
            try:
                maneuver_result = self.qr_detector.detect(tsf.frame)
                if maneuver_result is not None and maneuver_result.maneuver_name is not None:
                    logger.info(
                        "QR detected: maneuver=%s payload=%s bbox=%s",
                        maneuver_result.maneuver_name,
                        maneuver_result.payload,
                        maneuver_result.bbox,
                    )
            except Exception as exc:
                logger.error("QRDetector error: %s", exc)

        # ── Phase 3: Traffic-light detection ─────────────────────────────────
        traffic_light_result = None
        if self.traffic_light_detector:
            try:
                traffic_light_result = self.traffic_light_detector.detect(tsf.frame)
                if traffic_light_result is not None:
                    logger.info(
                        "Traffic light detected: state=%s confidence=%.3f bbox=%s",
                        traffic_light_result.state,
                        traffic_light_result.confidence,
                        traffic_light_result.bbox,
                    )
            except Exception as exc:
                logger.error("TrafficLightDetector error: %s", exc)

        # ── Phase 3: Motion detection ────────────────────────────────────────
        motion_result = None
        if self.motion_detector:
            try:
                motion_result = self.motion_detector.detect(tsf.frame)
                if motion_result is not None:
                    logger.info(
                        "Motion detected: moving=%s ratio=%.4f changed=%d/%d roi=%s",
                        motion_result.is_moving,
                        motion_result.pixel_change_ratio,
                        motion_result.changed_pixels,
                        motion_result.total_pixels,
                        motion_result.roi,
                    )
            except Exception as exc:
                logger.error("MotionDetector error: %s", exc)

        # ── Phase 3: Traffic-light + motion rule logging ────────────────────
        if traffic_light_result is not None and motion_result is not None:
            try:
                tl_state = getattr(traffic_light_result.state, "value", str(traffic_light_result.state))
                if tl_state == "red" and motion_result.is_moving:
                    logger.warning(
                        "Traffic rule check: RED + moving (ratio=%.4f)",
                        motion_result.pixel_change_ratio,
                    )
                elif tl_state == "green" and motion_result.is_moving:
                    logger.info(
                        "Traffic rule check: GREEN + moving (ratio=%.4f)",
                        motion_result.pixel_change_ratio,
                    )
            except Exception as exc:
                logger.error("Traffic rule check error: %s", exc)

        # ── Phase 4: Scoring ──────────────────────────────────────────────────
        # score_frame() → pure lateral math, returns None if not calibrated.
        # record_traffic_event() → tracks RED+moving streak, fires penalty at threshold.
        # add_frame_score() → appends to maneuver buffer for later aggregation.
        frame_score = None
        if self.scoring_engine and lane_result:
            try:
                frame_score = self.scoring_engine.score_frame(
                    lane_result.left_line,
                    lane_result.right_line,
                )
                self.scoring_engine.record_traffic_event(
                    traffic_light_result, motion_result
                )
                if frame_score is not None:
                    self.scoring_engine.add_frame_score(frame_score)
                    self._latest_score = frame_score  # Cache for debug overlay
                    
                    # Log explicitly every ~10 frames so we don't spam too hard
                    if self.frames_processed % 10 == 0:
                        logger.info("ScoringEngine: Phase 4 Score: %.1f/100 (Drift: %scm)", 
                                  frame_score.score, round(frame_score.error_cm, 1))

            except Exception as exc:
                logger.error("ScoringEngine error: %s", exc)

        # ── Phase 5: Test controller update ───────────────────────────────────
        # Note: TestController.update() only reacts to QR detections (maneuver_result).
        # It will process maneuver transitions even if scoring is not active.
        if self.test_controller:
            try:
                result = self.test_controller.update(
                    maneuver_result      = maneuver_result,
                    traffic_light_result = traffic_light_result,
                    motion_result        = motion_result,
                    lane_result          = lane_result,
                    frame_score          = frame_score,
                )
                if result is not None:
                    logger.info(
                        "TestController: run completed — candidate=%s passed=%s total=%.1f",
                        result.candidate_id, result.passed, result.total_score
                    )
            except Exception as exc:
                logger.error("TestController error: %s", exc)

        # ── Phase 6 stub: Dashboard push ──────────────────────────────────────
        # Emit only every Nth frame to reduce bandwidth and processing overhead
        # Default: every 3rd frame → 10fps becomes 3.3fps for dashboard (still smooth)
        if self.dashboard and (self.frames_processed % self.dashboard_emit_every_n == 0):
            try:
                display_frame = (
                    lane_result.raw_frame
                    if lane_result is not None
                    else tsf.frame
                )
                self.dashboard.emit_update(
                    frame                = display_frame,
                    frame_score          = frame_score,
                    maneuver_result      = maneuver_result,
                    traffic_light_result = traffic_light_result,
                    motion_result        = motion_result,
                )
            except Exception as exc:
                logger.error("Dashboard emit error: %s", exc)

    # ── Debug save ────────────────────────────────────────────────────────────

    def _debug_save(self, frame: np.ndarray, timestamp_ms: float, frame_score=None):
        """
        Save a frame as JPEG with metadata overlay.

        In Phase 1, this was the raw frame.
        In Phase 2+, this is typically the overlay frame (with lane lines, etc.).

        Filename: debug_frames/<timestamp_ms>.jpg
        Overlay (top-left): "ts=<ms>  q=<depth>  proc=<count>"  in green text.
        Overlay (row 2): Phase 4 Score.
        """
        try:
            annotated = frame.copy()
            label = (
                f"ts={timestamp_ms:.0f}ms  "
                f"q={self.frame_queue.size()}  "
                f"proc={self.frames_processed}"
            )
            cv2.putText(
                annotated,
                label,
                (5, 20),                    # top-left position
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,                       # font scale
                (0, 255, 0),                # green
                1,
                cv2.LINE_AA,
            )

            # Phase 4 Visual Overlays
            if frame_score is not None:
                score_label = f"Score: {frame_score.score:.1f}/100  Drift: {frame_score.error_cm:.1f}cm"
                cv2.putText(
                    annotated,
                    score_label,
                    (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255) if frame_score.score < 50 else (255, 255, 0), # cyan/yellow or red
                    1,
                    cv2.LINE_AA,
                )

            filename = os.path.join(
                DEBUG_FRAMES_DIR, f"{timestamp_ms:.0f}.jpg"
            )
            ok = cv2.imwrite(filename, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                logger.warning("imwrite failed for %s", filename)
        except Exception as exc:
            logger.error("_debug_save error: %s", exc)