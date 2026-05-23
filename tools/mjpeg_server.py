#!/usr/bin/env python3
"""
Synthetic MJPEG server: generates moving frames at 30fps and serves them
at /stream as multipart/x-mixed-replace so the backend can connect for testing.

Usage:
  python tools/mjpeg_server.py [port]

Default port: 8080
"""
from flask import Flask, Response
import cv2
import numpy as np
import time
import sys

app = Flask(__name__)
FPS = 30
W, H = 320, 240

def gen():
    x = 0
    dx = 4
    while True:
        # create a simple moving-pattern frame
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        # background gradient
        for i in range(H):
            color = int(255 * i / H)
            frame[i, :, :] = (color // 2, color, 255 - color)
        # moving rectangle
        cv2.rectangle(frame, (x, 60), (x + 40, 120), (0, 255, 0), -1)
        # timestamp text
        ts = time.time()
        cv2.putText(frame, f"t={ts:.2f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

        x += dx
        if x > W:
            x = -40
        time.sleep(1.0 / FPS)

@app.route('/stream')
def stream():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print(f"Starting synthetic MJPEG server on http://0.0.0.0:{port}/stream")
    app.run(host='0.0.0.0', port=port, threaded=True)
