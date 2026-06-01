"""
dashboard.py — Phase 6 Real-time WebSocket Dashboard

Bridges the synchronous computer-vision pipeline in QueueConsumer (Thread B)
with the asynchronous FastAPI / Uvicorn server using python-socketio.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Optional

import cv2
import numpy as np
import socketio

from config import FrameScore, ManeuverResult, TrafficLightResult, MotionResult

logger = logging.getLogger(__name__)

# Module-level Socket.IO server so main.py can wrap it via ASGI before startup.
sio_server = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

@sio_server.event
async def connect(sid, environ, auth):
    logger.info("Dashboard: Client connected (sid=%s)", sid)

@sio_server.event
async def disconnect(sid):
    logger.info("Dashboard: Client disconnected (sid=%s)", sid)

class Dashboard:
    """
    Manages the Socket.IO server and provides a thread-safe emit method.

    Parameters
    ----------
    loop : asyncio.AbstractEventLoop
        The running event loop from the main thread (usually uvicorn's loop).
    """

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self.sio = sio_server

    def emit_update(
        self,
        frame: np.ndarray,
        frame_score: Optional[FrameScore],
        maneuver_result: Optional[ManeuverResult],
        traffic_light_result: Optional[TrafficLightResult],
        motion_result: Optional[MotionResult],
    ) -> None:
        """
        Called BY THREAD B (QueueConsumer) every frame.
        Encodes the frame to JPEG Base64 and pushes the telemetry dictionary
        to the async event loop to be broadcasted to all connected web clients.
        """
        try:
            payload = self._build_payload(
                frame, frame_score, maneuver_result, traffic_light_result, motion_result
            )
            # Emit to all connected clients asynchronously
            asyncio.run_coroutine_threadsafe(
                self.sio.emit("telemetry", payload, to=None), self._loop
            )
        except Exception as exc:
            logger.error("Dashboard: Error emitting update: %s", exc)

    def _build_payload(
        self,
        frame: np.ndarray,
        frame_score: Optional[FrameScore],
        maneuver_result: Optional[ManeuverResult],
        traffic_light_result: Optional[TrafficLightResult],
        motion_result: Optional[MotionResult],
    ) -> dict:
        """Constructs the JSON-serialisable dictionary."""
        
        # 1. Encode frame to JPEG Base64 — ultra-low quality (40) for speed
        # At 10fps with this quality, payload is ~15-20KB per frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        success, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        if success:
            encoded_bytes = base64.b64encode(buffer) # type: ignore
            encoded_str = encoded_bytes.decode('utf-8')
            frame_b64 = f"data:image/jpeg;base64,{encoded_str}"
        else:
            frame_b64 = None
            logger.warning("Dashboard: Failed to encode JPEG frame")

        # 2. Extract metrics safely
        traffic_state = "none"
        if traffic_light_result and hasattr(traffic_light_result.state, 'value'):
            traffic_state = traffic_light_result.state.value
        elif traffic_light_result:
            traffic_state = str(traffic_light_result.state).lower()

        maneuver_name = None
        if maneuver_result and maneuver_result.maneuver_name:
            maneuver_name = maneuver_result.maneuver_name

        return {
            "frame_b64": frame_b64,
            
            # Phase 4 Scoring
            "score": round(frame_score.score, 1) if frame_score else None,
            "error_cm": round(frame_score.error_cm, 1) if frame_score else None,
            
            # Phase 3 Detections
            "maneuver": maneuver_name,
            "traffic_light": traffic_state,
            "is_moving": motion_result.is_moving if motion_result else False,
        }
