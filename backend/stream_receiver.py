"""
stream_receiver.py — Thread A: consumes the MJPEG stream from the ESP32-CAM.

Responsibilities (Phase 1):
  1. Open the ESP32 MJPEG stream with OpenCV.
  2. Decode every frame.
  3. Encode as JPEG and upload to MinIO (full-resolution recording).
  4. Every (CAMERA_FPS / QUEUE_SAMPLE_FPS) frames = every 3rd frame at 30 fps
     → copy + timestamp → push to FrameQueue for async processing.
  5. Always update latest_frame so the /video_feed HTTP endpoint can serve it.

OMITTED (future phases):
  - MinIO multipart upload for the full video file (currently uploads individual
    JPEG frames; a proper AVI/MP4 would use a VideoWriter or the MinIO multipart
    API — deferred to a dedicated storage phase).
  - Automatic reconnect with exponential backoff (stub _try_reconnect shows
    where this goes).
  - Hardware timestamp from ESP32 EXIF data.

STUB INTERFACE FOR PHASE 2+:
  - latest_frame is already set here; any downstream code just reads it.
  - frame_queue is already populated here; QueueConsumer just needs to plug in.
"""

import io
import logging
import threading
import time

import cv2
import numpy as np

from config import (
    CAMERA_FPS,
    QUEUE_SAMPLE_FPS,
    MINIO_BUCKET,
    TimestampedFrame,
)

logger = logging.getLogger(__name__)


class StreamReceiver(threading.Thread):
    """
    Daemon thread — starts at backend startup, exits when the main process exits.

    Parameters
    ----------
    stream_url  : Full URL of the ESP32 MJPEG stream, e.g. "http://192.168.1.xxx/stream"
    frame_queue : FrameQueue instance shared with QueueConsumer (Thread B).
    minio_client: Initialised Minio() client, or None to skip MinIO recording.
    """

    def __init__(self, stream_url: str, frame_queue, minio_client=None):
        super().__init__(daemon=True, name="StreamReceiver")
        self.stream_url   = stream_url
        self.frame_queue  = frame_queue
        self.minio_client = minio_client

        # Thread-safe latest frame — read by /video_feed endpoint on main thread
        self._frame_lock  = threading.Lock()
        self._latest: np.ndarray | None = None

        self._stop_event   = threading.Event()
        # When set, the receiver is paused (no new frames are pushed to queue)
        self._paused_event = threading.Event()
        self._frame_index  = 0
        # e.g. CAMERA_FPS=30, QUEUE_SAMPLE_FPS=10 → sample every 3rd frame
        self._sample_every = max(1, CAMERA_FPS // QUEUE_SAMPLE_FPS)

        # Stats for dashboard / logging
        self.frames_received = 0
        self.frames_to_queue = 0
        self.minio_errors    = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def latest_frame(self) -> np.ndarray | None:
        """Returns the most recent decoded frame (thread-safe copy)."""
        with self._frame_lock:
            return self._latest

    def stop(self):
        """Signal the thread to exit after the current frame."""
        self._stop_event.set()

    def pause(self):
        """Pause sampling and pushing frames to the processing queue."""
        self._paused_event.set()

    def resume(self):
        """Resume sampling and pushing frames to the processing queue."""
        self._paused_event.clear()

    @property
    def is_paused(self) -> bool:
        return self._paused_event.is_set()

    # ── Thread body ───────────────────────────────────────────────────────────

    def run(self):
        logger.info("StreamReceiver starting — URL: %s", self.stream_url)
        logger.info(
            "Sampling every %d frame(s) → ~%d fps to processing queue",
            self._sample_every, QUEUE_SAMPLE_FPS,
        )

        cap = self._open_capture()
        if cap is None:
            logger.error("Could not open stream; StreamReceiver exiting.")
            return

        consecutive_failures = 0
        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    logger.warning(
                        "Frame read failed (ret=%s, frame=%s) — retrying...",
                        ret,
                        "None" if frame is None else f"shape={frame.shape}",
                    )
                if consecutive_failures >= 20:
                    logger.error(
                        "20+ consecutive frame read failures — stream may be disconnected. "
                        "Attempting to reconnect..."
                    )
                    cap.release()
                    time.sleep(1.0)
                    cap = self._open_capture()
                    if cap is None:
                        logger.error("Reconnection failed; exiting.")
                        return
                    consecutive_failures = 0
                else:
                    time.sleep(0.05)
                continue
            
            consecutive_failures = 0

            ts_ms = time.monotonic_ns() / 1_000_000
            self._frame_index += 1
            self.frames_received += 1

            # ── Update latest frame (for /video_feed live display) ────────────
            with self._frame_lock:
                self._latest = frame   # no copy needed; we won't mutate it here

            # ── Record to MinIO (every frame, full resolution) ────────────────
            if self.minio_client is not None:
                self._record_to_minio(frame, ts_ms)

            # ── Sample at QUEUE_SAMPLE_FPS → push to processing queue ─────────
            # If paused, skip pushing sampled frames (but continue updating latest_frame)
            if not self._paused_event.is_set() and (self._frame_index % self._sample_every == 0):
                tsf = TimestampedFrame(
                    frame=frame.copy(),   # copy so Thread B owns this memory
                    timestamp_ms=ts_ms,
                )
                pushed = self.frame_queue.push(tsf)
                if pushed:
                    self.frames_to_queue += 1

            # ── Periodic stats log ────────────────────────────────────────────
            if self.frames_received % 300 == 0:   # every ~10 s at 30 fps
                logger.info(
                    "StreamReceiver stats — received=%d  queued=%d  "
                    "queue_depth=%d  minio_errors=%d",
                    self.frames_received,
                    self.frames_to_queue,
                    self.frame_queue.size(),
                    self.minio_errors,
                )

        cap.release()
        logger.info("StreamReceiver stopped.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _open_capture(self) -> cv2.VideoCapture | None:
        """Opens the MJPEG stream with explicit codec flags for compatibility."""
        cap = cv2.VideoCapture(self.stream_url)
        
        # Set timeouts
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5_000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000)
        
        # Force MJPEG codec (0x47504A4D = 'MJPG' in fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Set buffer size (prevent frame drops)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if not cap.isOpened():
            logger.error("cv2.VideoCapture could not open: %s", self.stream_url)
            return None
        
        # Log actual stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Stream opened successfully — resolution=%dx%d @ %.1f fps",
            width, height, fps,
        )
        return cap

    def _record_to_minio(self, frame: np.ndarray, ts_ms: float):
        """
        Encode frame as JPEG and PUT to MinIO under recordings/<ts_ms>.jpg.

        NOTE (Phase 1 limitation):
          This uploads individual frames, not a video container.  The bucket will
          accumulate thousands of small JPEG objects.  A production implementation
          would use a background MinIO-writer thread and multipart upload to build
          a proper video file.  That is marked as a TODO for the storage phase.
        """
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            if not ok:
                return

            data = buf.tobytes()
            key  = f"frames/{ts_ms:.0f}.jpg"

            self.minio_client.put_object(
                bucket_name  = MINIO_BUCKET,
                object_name  = key,
                data         = io.BytesIO(data),
                length       = len(data),
                content_type = "image/jpeg",
            )
        except Exception as exc:
            self.minio_errors += 1
            if self.minio_errors % 50 == 1:
                logger.warning("MinIO upload error (total: %d): %s",
                               self.minio_errors, exc)

    # ── STUB: reconnect logic (Phase 1 — not implemented) ────────────────────
    def _try_reconnect(self) -> cv2.VideoCapture | None:
        """
        STUB — Phase 1 exits on stream failure.
        Phase 2+: implement exponential backoff + re-open here.
        """
        raise NotImplementedError("Reconnect not yet implemented")