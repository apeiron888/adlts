"""
main.py — FastAPI entry point for the lane-scoring backend.

Phase 1 (ACTIVE):
  • Start MinIO client (connect to S3-compatible storage)
  • Create frame queue and threads (StreamReceiver + QueueConsumer)
  • Expose HTTP endpoints: /video_feed (MJPEG), /stats (JSON), /health
  • Graceful shutdown on interrupt

Phase 2+ (STUB):
  • Initialize lane_detector, sign_detector, scoring_engine, test_controller
  • Pass them to QueueConsumer in the constructor
  • Phase 2: Uncomment lane_detector=lane_detector below and implement LaneDetector
"""

import logging
import signal
import sys
import time

import cv2
import numpy as np
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from minio import Minio

from config import (
    MINIO_ENDPOINT,
    MINIO_ACCESS,
    MINIO_SECRET,
    MINIO_BUCKET,
    MINIO_SECURE,
    ESP32_STREAM_URL,
)
from frame_queue import FrameQueue
from stream_receiver import StreamReceiver
from queue_consumer import QueueConsumer

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

# ─── Startup & shutdown ──────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    """Initialize threads and services."""
    global _frame_queue, _stream_receiver, _queue_consumer, _minio_client

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

    # Start QueueConsumer (Thread B)
    # Phase 2+: pass lane_detector, sign_detector, scoring_engine, etc.
    _queue_consumer = QueueConsumer(
        frame_queue=_frame_queue,
        # lane_detector=LaneDetector(),  # Phase 2 — uncomment when ready
        # sign_detector=SignDetector(),  # Phase 3
        # scoring_engine=ScoringEngine(),  # Phase 4
        # test_controller=TestController(),  # Phase 5
        # dashboard=Dashboard(),  # Phase 6
        save_debug=True,  # Enable debug frame saving in Phase 1
    )
    _queue_consumer.start()
    logger.info("QueueConsumer thread started")

    logger.info("=" * 70)
    logger.info("Startup complete — streaming active")
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
        },
        "queue": {
            "depth": _frame_queue.size(),
            "dropped": _frame_queue.dropped,
        },
        "queue_consumer": {
            "frames_processed": _queue_consumer.frames_processed,
        },
    }


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
