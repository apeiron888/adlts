"""
main.py — FastAPI entry point for the lane-scoring backend.

Phase 1 (ACTIVE):
  • Start MinIO client (connect to S3-compatible storage)
  • Create frame queue and threads (StreamReceiver + QueueConsumer)
  • Expose HTTP endpoints: /video_feed (MJPEG), /stats (JSON), /health
  • Graceful shutdown on interrupt

Phase 2+ (STUB):
    • Initialize lane_detector, qr_detector, traffic_light_detector, motion_detector,
        scoring_engine, test_controller
  • Pass them to QueueConsumer in the constructor
  • Phase 2: Uncomment lane_detector=lane_detector below and implement LaneDetector
"""

import logging
import os
import signal
import sys
import time
import asyncio

import cv2
import numpy as np
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from minio import Minio
import socketio

from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS,
    MINIO_SECRET,
    MINIO_BUCKET,
    MINIO_SECURE,
    ESP32_STREAM_URL,
    LANE_WIDTH_CM,
    TRAFFIC_LIGHT_MODEL_PATH,
)
from frame_queue import FrameQueue
from stream_receiver import StreamReceiver
from queue_consumer import QueueConsumer
from lane_detector import LaneDetector
from qr_detector import QRDetector
from traffic_light_detector import TrafficLightDetector
from motion_detector import MotionDetector
from scoring_engine import ScoringEngine
from test_controller import TestController
from dashboard import Dashboard, sio_server

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Global state (initialized at startup) ──────────────────────────────────

app = FastAPI(title="Lane Scoring Backend")

_frame_queue: FrameQueue | None = None
_stream_receiver: StreamReceiver | None = None
_queue_consumer: QueueConsumer | None = None
_minio_client: Minio | None = None
_lane_detector: LaneDetector | None = None
_qr_detector = None
_traffic_light_detector = None
_motion_detector = None
_scoring_engine: ScoringEngine | None = None
_test_controller: TestController | None = None
_dashboard: Dashboard | None = None

# ─── Startup & shutdown ──────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    """Initialize threads and services."""
    global _frame_queue, _stream_receiver, _queue_consumer, _minio_client
    global _lane_detector, _qr_detector, _traffic_light_detector, _motion_detector
    global _scoring_engine, _test_controller, _dashboard

    logger.info("=" * 70)
    logger.info("Backend startup sequence starting...")
    logger.info("=" * 70)

    # Initialize MinIO client
    try:
        _minio_client = Minio(
            endpoint=MINIO_ENDPOINT,
            access_key=MINIO_ACCESS,
            secret_key=MINIO_SECRET,
            secure=MINIO_SECURE,
        )
        # Ensure bucket exists
        if not _minio_client.bucket_exists(MINIO_BUCKET):
            _minio_client.make_bucket(MINIO_BUCKET)
            logger.info(f"Created MinIO bucket: {MINIO_BUCKET}")
        else:
            logger.info(f"MinIO bucket {MINIO_BUCKET} already exists")
    except Exception as e:
        logger.error(f"MinIO initialization failed: {e}")
        logger.warning("Continuing without MinIO recording...")
        _minio_client = None

    # Create shared queue
    _frame_queue = FrameQueue()

    # Start StreamReceiver (Thread A)
    _stream_receiver = StreamReceiver(
        stream_url=ESP32_STREAM_URL,
        frame_queue=_frame_queue,
        minio_client=_minio_client,
    )
    _stream_receiver.start()
    logger.info("StreamReceiver thread started")

    # Initialize lane detector (Phase 2)
    _lane_detector = LaneDetector()
    logger.info("LaneDetector initialized")

    # Initialize Phase 3 detectors (REAL mode only)
    _qr_detector = QRDetector(cooldown_s=3.0)
    _traffic_light_detector = TrafficLightDetector(model_path=TRAFFIC_LIGHT_MODEL_PATH)
    _motion_detector = MotionDetector(movement_threshold_ratio=0.01)
    logger.info(
        "Phase 3 detectors initialized (REAL mode) — QRDetector, TrafficLightDetector, MotionDetector"
    )

    # Initialize Phase 4 scoring engine
    _scoring_engine = ScoringEngine(lane_detector=_lane_detector)
    logger.info("ScoringEngine initialized")

    # Initialize Phase 5 test controller
    _test_controller = TestController(scoring_engine=_scoring_engine)
    logger.info("TestController initialized")

    # Initialize Phase 6 dashboard
    _dashboard = Dashboard(loop=asyncio.get_running_loop())
    logger.info("Dashboard initialized — Socket.IO ready at /socket.io")

    # Start QueueConsumer (Thread B)
    _queue_consumer = QueueConsumer(
        frame_queue=_frame_queue,
        lane_detector=_lane_detector,
        qr_detector=_qr_detector,
        traffic_light_detector=_traffic_light_detector,
        motion_detector=_motion_detector,
        scoring_engine=_scoring_engine,   # Phase 4 — active
        test_controller=_test_controller,  # Phase 5 — active
        dashboard=_dashboard,              # Phase 6 — active
        save_debug=True,
        dashboard_emit_every_n=3,  # Emit to dashboard every 3rd frame → reduces bandwidth 3x
    )
    _queue_consumer.start()
    logger.info("QueueConsumer thread started")

    logger.info("=" * 70)
    logger.info("Startup complete — streaming and lane detection active")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully stop threads."""
    global _stream_receiver, _queue_consumer

    logger.info("Shutdown signal received")
    if _stream_receiver:
        _stream_receiver.stop()
        _stream_receiver.join(timeout=2)
    if _queue_consumer:
        _queue_consumer.stop()
        _queue_consumer.join(timeout=5)
    logger.info("Shutdown complete")


# ─── HTTP Endpoints ──────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "receiver_running": _stream_receiver is not None and _stream_receiver.is_alive(),
        "consumer_running": _queue_consumer is not None and _queue_consumer.is_alive(),
        "receiver_paused": _stream_receiver is not None and _stream_receiver.is_paused if _stream_receiver is not None else False,
        "lane_detector_ready": _lane_detector is not None,
    }


@app.post("/calibrate")
async def calibrate():
    """
    Calibration endpoint for Phase 2 lane detection.
    
    Place the car dead centre on a straight lane, then POST to this endpoint.
    It captures the current frame, detects both lanes, and computes pixels_per_cm.
    
    Returns:
        - message: "Calibration done"
        - pixels_per_cm: Conversion factor for later use by ScoringEngine
        - lane_width_px: Pixel width of the lane at the bottom
    """
    if not _lane_detector:
        return {"error": "LaneDetector not initialized"}

    if _stream_receiver is None or _stream_receiver.latest_frame is None:
        return {"error": "No frame available from stream"}

    result = _lane_detector.detect(_stream_receiver.latest_frame)

    if result.left_line and result.right_line:
        _lane_detector.calibrate(result.left_line, result.right_line, LANE_WIDTH_CM)
        logger.info(
            f"Calibration done — pixels_per_cm={_lane_detector.pixels_per_cm:.2f}, "
            f"lane_width_px={_lane_detector.lane_width_px}"
        )
        return {
            "message": "Calibration done",
            "pixels_per_cm": _lane_detector.pixels_per_cm,
            "lane_width_px": _lane_detector.lane_width_px,
        }
    else:
        return {
            "error": "Could not find both lanes in current frame",
            "left_line": result.left_line,
            "right_line": result.right_line,
        }


@app.get("/stats")
async def stats():
    """Return current pipeline statistics."""
    if not _stream_receiver or not _queue_consumer or not _frame_queue:
        return {"error": "Backend not initialized"}

    return {
        "stream_receiver": {
            "frames_received": _stream_receiver.frames_received,
            "frames_to_queue": _stream_receiver.frames_to_queue,
            "minio_errors": _stream_receiver.minio_errors,
            "paused": _stream_receiver.is_paused if _stream_receiver is not None else False,
        },
        "queue": {
            "depth": _frame_queue.size(),
            "dropped": _frame_queue.dropped,
        },
        "queue_consumer": {
            "frames_processed": _queue_consumer.frames_processed,
        },
    }


@app.get("/score")
async def score():
    """
    Return the latest per-frame scoring state from the ScoringEngine.

    Useful for live monitoring: poll this endpoint to see the car's current
    lateral score, drift, violations, and how many frames have been buffered
    for the active maneuver.

    Returns 'calibrated: false' until POST /calibrate has been called.
    """
    if not _scoring_engine:
        return {"error": "ScoringEngine not initialized"}

    latest = getattr(_queue_consumer, "_latest_score", None) if _queue_consumer else None

    return {
        "calibrated": _scoring_engine.is_calibrated,
        "pixels_per_cm": _scoring_engine.pixels_per_cm,
        "latest_frame": {
            "score": latest.score if latest else None,            # 0–100
            "error_cm": latest.error_cm if latest else None,      # physical drift
            "error_px": latest.error_px if latest else None,      # pixel drift
        },
        "active_maneuver_buffer": {
            "frames_scored": len(_scoring_engine._frame_scores),
            "mean_so_far": round(
                float(sum(_scoring_engine._frame_scores) / len(_scoring_engine._frame_scores)), 2
            ) if _scoring_engine._frame_scores else None,
        },
        "violations": {
            "count": _scoring_engine._violations,
            "penalty_pts": _scoring_engine._penalty,
            "red_moving_streak": _scoring_engine._red_moving_streak,
        },
    }


@app.post("/score/maneuver")
async def score_maneuver(name: str = "unnamed"):
    """
    Finalise the current maneuver and return its aggregated ManeuverScore.

    This is normally triggered automatically by a QR code in Phase 5.
    Call it manually here to close out the current maneuver buffer and see
    the trimmed-mean score + penalty.

    Query param: ?name=straight_1   (default: 'unnamed')
    """
    if not _scoring_engine:
        return {"error": "ScoringEngine not initialized"}
    if not _scoring_engine.is_calibrated:
        return {"error": "Not yet calibrated — call POST /calibrate first"}

    ms = _scoring_engine.aggregate_maneuver(name)
    return {
        "maneuver": ms.name,
        "raw_score": ms.raw_score,
        "penalty": ms.penalty,
        "final_score": ms.final_score,
        "frames": ms.frame_count,
        "violations": ms.violations,
    }



@app.post("/test/start")
async def test_start(candidate_id: str = "unknown"):
    """
    Start a new driving test.

    The car must be calibrated first (POST /calibrate).
    Query param: ?candidate_id=cand_001
    """
    if not _test_controller:
        return {"error": "TestController not initialized"}
    if not _scoring_engine or not _scoring_engine.is_calibrated:
        return {"error": "Not calibrated — call POST /calibrate first"}
    try:
        _test_controller.start_test(candidate_id)
        return {
            "message": "Test started",
            "candidate_id": candidate_id,
            "first_maneuver": _test_controller.current_maneuver,
        }
    except RuntimeError as exc:
        return {"error": str(exc)}


@app.get("/test/status")
async def test_status():
    """Return the current state of the active test."""
    if not _test_controller:
        return {"error": "TestController not initialized"}
    return _test_controller.progress


@app.post("/test/abort")
async def test_abort():
    """Abort the running test and reset to IDLE."""
    if not _test_controller:
        return {"error": "TestController not initialized"}
    _test_controller.abort()
    return {"message": "Test aborted", "state": _test_controller.state.value}



@app.post("/stream/stop")
async def stream_stop():
    """Pause the StreamReceiver (stop pushing new frames to the processing queue)."""
    if not _stream_receiver:
        return {"error": "StreamReceiver not initialized"}
    _stream_receiver.pause()
    return {"message": "StreamReceiver paused", "paused": _stream_receiver.is_paused}


@app.post("/stream/start")
async def stream_start():
    """Resume the StreamReceiver (allow pushing sampled frames again)."""
    if not _stream_receiver:
        return {"error": "StreamReceiver not initialized"}
    _stream_receiver.resume()
    return {"message": "StreamReceiver resumed", "paused": _stream_receiver.is_paused}


def _generate_mjpeg():
    """Generator for MJPEG stream — yields one frame at a time."""
    while True:
        if _stream_receiver is None:
            time.sleep(0.1)
            continue

        frame = _stream_receiver.latest_frame
        if frame is None:
            time.sleep(0.1)
            continue

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = buffer.tobytes()

        # MJPEG boundary format
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" +
            data + b"\r\n"
        )

        time.sleep(0.033)  # ~30 fps


@app.get("/video_feed")
async def video_feed():
    """Live MJPEG stream endpoint."""
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/dashboard", response_class=None)
async def dashboard_page():
    """Serve the live dashboard HTML over HTTP (avoids file:// WebSocket restrictions)."""
    import pathlib
    from fastapi.responses import HTMLResponse
    html_path = pathlib.Path(__file__).parent / "test_client.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


# ─── ASGI Wrapper for Socket.IO ──────────────────────────────────────────────

# Wrap the entire FastAPI application with Socket.IO so it correctly intercepts
# /socket.io paths before passing everything else to FastAPI.
app = socketio.ASGIApp(sio_server, other_asgi_app=app)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server on 0.0.0.0:8000")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
