#!/usr/bin/env python3
"""
Minimal MJPEG HTTP server using only standard library.
Serves synthetic frames at /stream at 30 fps.
"""
import io, time
from http import server
import socketserver
import threading
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    # If Pillow not available, generate raw numpy frames via simple approach
    Image = None

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
FPS = 30
W, H = 320, 240

class MJPEGHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/stream':
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not found')
            return
        self.send_response(200)
        self.send_header('Age', '0')
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        x = 0
        dx = 4
        try:
            while True:
                # create RGB image bytes
                if Image:
                    img = Image.new('RGB', (W, H))
                    draw = ImageDraw.Draw(img)
                    # gradient
                    for i in range(H):
                        color = int(255 * i / H)
                        draw.line([(0, i), (W, i)], fill=(color//2, color, 255-color))
                    draw.rectangle((x, 60, x+40, 120), fill=(0,255,0))
                    draw.text((5,5), f"t={time.time():.2f}", fill=(255,255,255))
                    bio = io.BytesIO()
                    img.save(bio, format='JPEG', quality=80)
                    frame = bio.getvalue()
                else:
                    # fallback: create simple RGB pattern
                    import numpy as np
                    import cv2
                    frame_np = np.zeros((H, W, 3), dtype='uint8')
                    for i in range(H):
                        color = int(255 * i / H)
                        frame_np[i,:,:] = (color//2, color, 255-color)
                    cv2.rectangle(frame_np, (x,60), (x+40,120), (0,255,0), -1)
                    cv2.putText(frame_np, f"t={time.time():.2f}", (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
                    ret, buf = cv2.imencode('.jpg', frame_np, [cv2.IMWRITE_JPEG_QUALITY,80])
                    frame = buf.tobytes()

                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(1.0 / FPS)
                x += dx
                if x > W:
                    x = -40
        except (BrokenPipeError, ConnectionResetError):
            return

class ThreadedHTTPServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True

if __name__ == '__main__':
    httpd = ThreadedHTTPServer(('0.0.0.0', PORT), MJPEGHandler)
    print(f"Serving synthetic MJPEG on http://0.0.0.0:{PORT}/stream")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
        print('Server stopped')
