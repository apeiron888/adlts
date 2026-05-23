"""
queue_consumer.py — Thread B: pops frames from FrameQueue and processes them.

Phase 1  (ACTIVE):
  • Saves every popped frame to debug_frames/<timestamp_ms>.jpg so you can
    visually confirm the queue is working.
  • Overlays timestamp and queue depth on the saved image.
  • Logs one line per frame to the console.

Phase 2+ (STUB — wired but inactive until the component is passed in):
  • LaneDetector.detect()     → LaneResult
  • SignDetector.detect()     → SignResult
  • ScoringEngine.score_frame()→ FrameScore
  • TestController.update()
  • Dashboard.emit_update()

To activate Phase 2, simply pass lane_detector=LaneDetector() to the
constructor.  No other change needed.  Same pattern for each later phase.

OMITTED (future phases):
  - Result persistence to MinIO / database
  - Overlay rendering pipeline (done in LaneDetector / SignDetector)
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
    sign_detector   : SignDetector instance, or None (Phase 3).
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
        sign_detector=None,
        scoring_engine=None,
        test_controller=None,
        dashboard=None,
        save_debug: bool = True,
    ):
        super().__init__(daemon=True, name="QueueConsumer")
        self.frame_queue     = frame_queue
        self.lane_detector   = lane_detector    # None → Phase 2 not active
        self.sign_detector   = sign_detector    # None → Phase 3 not active
        self.scoring_engine  = scoring_engine   # None → Phase 4 not active
        self.test_controller = test_controller  # None → Phase 5 not active
        self.dashboard       = dashboard        # None → Phase 6 not active
        self.save_debug      = save_debug

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

        # ── Phase 1+2: Save frame to disk ────────────────────────────────────
        # In Phase 2, we save the lane overlay if available; else save raw frame.
        # This lets us visually confirm the lane detection is working.
        if self.save_debug:
            frame_to_save = lane_result.raw_frame if lane_result else tsf.frame
            self._debug_save(frame_to_save, tsf.timestamp_ms)

        # ── Phase 3 stub: Sign detection ──────────────────────────────────────
        sign_result = None
        if self.sign_detector:
            try:
                sign_result = self.sign_detector.detect(tsf.frame)
            except Exception as exc:
                logger.error("SignDetector error: %s", exc)

        # ── Phase 4 stub: Scoring ─────────────────────────────────────────────
        frame_score = None
        if self.scoring_engine and lane_result:
            try:
                frame_score = self.scoring_engine.score_frame(
                    tsf.frame,
                    lane_result.left_line,
                    lane_result.right_line,
                )
            except Exception as exc:
                logger.error("ScoringEngine error: %s", exc)

        # ── Phase 5 stub: Test controller update ──────────────────────────────
        if self.test_controller and frame_score is not None and sign_result is not None:
            try:
                self.test_controller.update(
                    sign_id     = sign_result.sign_id,
                    lane_result = lane_result,
                    frame_score = frame_score.score,
                )
            except Exception as exc:
                logger.error("TestController error: %s", exc)

        # ── Phase 6 stub: Dashboard push ──────────────────────────────────────
        if self.dashboard:
            try:
                display_frame = (
                    lane_result.raw_frame
                    if lane_result is not None
                    else tsf.frame
                )
                self.dashboard.emit_update(
                    frame       = display_frame,
                    frame_score = frame_score,
                    sign_result = sign_result,
                )
            except Exception as exc:
                logger.error("Dashboard emit error: %s", exc)

    # ── Debug save ────────────────────────────────────────────────────────────

    def _debug_save(self, frame: np.ndarray, timestamp_ms: float):
        """
        Save a frame as JPEG with metadata overlay.

        In Phase 1, this was the raw frame.
        In Phase 2+, this is typically the overlay frame (with lane lines, etc.).

        Filename: debug_frames/<timestamp_ms>.jpg
        Overlay (top-left): "ts=<ms>  q=<depth>  proc=<count>"  in green text.
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

            filename = os.path.join(
                DEBUG_FRAMES_DIR, f"{timestamp_ms:.0f}.jpg"
            )
            ok = cv2.imwrite(filename, annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                logger.warning("imwrite failed for %s", filename)
        except Exception as exc:
            logger.error("_debug_save error: %s", exc)