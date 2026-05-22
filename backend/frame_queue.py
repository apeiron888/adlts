"""
frame_queue.py — Thread-safe FIFO between StreamReceiver (Thread A)
                 and QueueConsumer (Thread B).

Design decisions:
  • Non-blocking push: if the queue is full, the oldest frame is silently
    dropped (newest-wins back-pressure).  Processing must never stall capture.
  • Blocking pop with timeout: consumer sleeps rather than busy-loops.
  • size() is approximate (queue.Queue.qsize() is not guaranteed atomic) but
    good enough for dashboard monitoring and logging.
"""

import queue
import logging
from config import TimestampedFrame, QUEUE_MAX_SIZE

logger = logging.getLogger(__name__)


class FrameQueue:
    """
    Thin wrapper around queue.Queue that adds:
      - Non-blocking push with drop-on-full semantics.
      - A size() convenience method.
      - Logging when frames are dropped (useful during tuning).
    """

    def __init__(self, maxsize: int = QUEUE_MAX_SIZE):
        self._q: queue.Queue[TimestampedFrame] = queue.Queue(maxsize=maxsize)
        self._dropped = 0   # cumulative dropped frame counter

    # ── Producer side (Thread A) ──────────────────────────────────────────────

    def push(self, tsf: TimestampedFrame) -> bool:
        """
        Non-blocking enqueue.
        Returns True on success, False if queue was full (frame dropped).
        """
        try:
            self._q.put_nowait(tsf)
            return True
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:  # log every 100th drop so we don't spam
                logger.warning(
                    "FrameQueue full (%d slots) — frame dropped "
                    "(total dropped: %d)",
                    self._q.maxsize, self._dropped,
                )
            return False

    # ── Consumer side (Thread B) ──────────────────────────────────────────────

    def pop(self, timeout: float = 1.0) -> TimestampedFrame | None:
        """
        Blocking dequeue with timeout.
        Returns the oldest TimestampedFrame, or None if timeout expires.
        """
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Monitoring ────────────────────────────────────────────────────────────

    def size(self) -> int:
        """Approximate current number of frames waiting in the queue."""
        return self._q.qsize()

    @property
    def dropped(self) -> int:
        """Total frames dropped since creation (for dashboard display)."""
        return self._dropped