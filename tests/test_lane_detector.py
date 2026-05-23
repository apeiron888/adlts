"""
tests/test_lane_detector.py — Phase 2 Lane Detector Testing

This test script loads images or videos and feeds them through the LaneDetector.
It displays the overlay (green lane lines + red center marker) for manual verification.

Usage:
  python tests/test_lane_detector.py path/to/image.jpg
  python tests/test_lane_detector.py path/to/video.mp4
"""

import sys
import os
import cv2
import numpy as np

# Allow importing from backend
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from lane_detector import LaneDetector
from config import LaneResult


def test_with_image(image_path: str):
    """Load a single image and display the lane detection overlay."""
    print(f"Loading image: {image_path}")
    detector = LaneDetector()
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"ERROR: Could not read {image_path}")
        return False
    
    print(f"Image shape: {frame.shape}")
    result = detector.detect(frame)
    
    print(f"Detection result:")
    print(f"  left_line:  {result.left_line}")
    print(f"  right_line: {result.right_line}")
    print(f"  centre_x:   {result.centre_x}")
    
    print("\nDisplaying overlay (press any key to close)...")
    cv2.imshow("Lane Detection Overlay", result.raw_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True


def test_with_video(video_path: str):
    """Load a video and display lane detection on each frame."""
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return False
    
    detector = LaneDetector()
    frame_count = 0
    
    print("Playing video (press 'q' to quit, 'space' to pause)...")
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print(f"\nEnd of video — processed {frame_count} frames")
                break
            
            frame_count += 1
            result = detector.detect(frame)
            
            # Add frame counter to overlay
            cv2.putText(
                result.raw_frame,
                f"Frame {frame_count}",
                (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            
            if frame_count % 10 == 0:
                print(f"  Frame {frame_count}: left={result.left_line}, "
                      f"right={result.right_line}, centre={result.centre_x}")
        
        cv2.imshow("Lane Detection - Video", result.raw_frame)
        key = cv2.waitKey(33) & 0xFF  # ~30 fps
        
        if key == ord('q'):
            print("Quit requested")
            break
        elif key == ord(' '):
            paused = not paused
            status = "PAUSED" if paused else "PLAYING"
            print(f"[{status}]")
    
    cap.release()
    cv2.destroyAllWindows()
    return True


def test_single_lane_fallback(image_path: str):
    """
    Test the single-lane fallback by:
    1. Loading an image with both lanes visible
    2. Running calibration
    3. Artificially masking one side of the image
    4. Running detection again to see fallback in action
    """
    print(f"\n{'='*70}")
    print("SINGLE-LANE FALLBACK TEST")
    print(f"{'='*70}")
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Could not read {image_path}")
        return False
    
    h, w = frame.shape[:2]
    detector = LaneDetector()
    
    # Step 1: Detect on original frame
    print("\n[Step 1] Detecting on original frame (calibration)...")
    result1 = detector.detect(frame)
    print(f"  left_line:  {result1.left_line}")
    print(f"  right_line: {result1.right_line}")
    
    if not (result1.left_line and result1.right_line):
        print("  ERROR: Both lanes must be visible for this test")
        return False
    
    # Step 2: Calibrate
    print("\n[Step 2] Running calibration...")
    detector.calibrate(result1.left_line, result1.right_line, lane_width_cm=30.0)
    print(f"  pixels_per_cm: {detector.pixels_per_cm}")
    print(f"  lane_width_px: {detector.lane_width_px}")
    
    # Step 3: Mask right half of image (hide right lane)
    print("\n[Step 3] Masking right half of image and re-detecting...")
    masked_frame = frame.copy()
    masked_frame[:, w//2:] = 0  # Black out right half
    result2 = detector.detect(masked_frame)
    print(f"  left_line:  {result2.left_line}")
    print(f"  right_line (fallback): {result2.right_line}")
    print(f"  centre_x:   {result2.centre_x}")
    
    # Display results side-by-side
    print("\n[Step 4] Displaying results...")
    print("  Left: Original (both lanes)")
    print("  Right: With right half masked (showing fallback)")
    
    combined = np.hstack([result1.raw_frame, result2.raw_frame])
    cv2.imshow("Lane Detection - Fallback Test", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True


def main():
    if len(sys.argv) < 2:
        print("USAGE:")
        print("  python tests/test_lane_detector.py <image_or_video_path>")
        print("  python tests/test_lane_detector.py --fallback <image_path>")
        print("\nExamples:")
        print("  python tests/test_lane_detector.py sample.jpg")
        print("  python tests/test_lane_detector.py track_video.mp4")
        print("  python tests/test_lane_detector.py --fallback calibration_frame.jpg")
        return 1
    
    if sys.argv[1] == "--fallback":
        if len(sys.argv) < 3:
            print("ERROR: --fallback requires an image path")
            return 1
        success = test_single_lane_fallback(sys.argv[2])
    else:
        path = sys.argv[1]
        
        if not os.path.exists(path):
            print(f"ERROR: Path does not exist: {path}")
            return 1
        
        # Detect file type by extension
        lower_path = path.lower()
        is_video = lower_path.endswith(('.mp4', '.avi', '.mov', '.mkv', '.mjpeg', '.flv', '.wmv'))
        
        if is_video:
            success = test_with_video(path)
        else:
            success = test_with_image(path)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
