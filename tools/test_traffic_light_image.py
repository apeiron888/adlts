#!/usr/bin/env python3
"""
Test traffic light detector on a static image.
Usage: python test_traffic_light_image.py <image_path>
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import cv2
import logging
from traffic_light_detector import TrafficLightDetector
from config import TRAFFIC_LIGHT_MODEL_PATH, TrafficLightState

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_image(image_path: str) -> None:
    """Test traffic light detection on a single image."""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Failed to read image: {image_path}")
        return
    
    print(f"✓ Loaded image: {image_path}")
    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    # Initialize detector
    print(f"\nInitializing TrafficLightDetector with model: {TRAFFIC_LIGHT_MODEL_PATH}")
    detector = TrafficLightDetector(model_path=TRAFFIC_LIGHT_MODEL_PATH)
    
    if detector._model is None:
        print("❌ YOLO model failed to load!")
        return
    
    # Run detection
    print("\nRunning detection...")
    result = detector.detect(frame)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Detection Result:")
    print(f"{'='*60}")
    print(f"State:       {result.state.name}")
    print(f"Confidence:  {result.confidence:.3f}")
    print(f"Bbox:        {result.bbox}")
    print(f"{'='*60}")
    
    # Draw on image if detection found
    if result.bbox:
        x1, y1, x2, y2 = result.bbox
        color = {
            TrafficLightState.RED: (0, 0, 255),     # Blue in BGR
            TrafficLightState.GREEN: (0, 255, 0),   # Green in BGR
            TrafficLightState.NONE: (128, 128, 128) # Gray
        }.get(result.state, (128, 128, 128))
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{result.state.name} ({result.confidence:.2f})"
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        
        # Save output
        output_path = image_path.replace(".jpg", "_detected.jpg").replace(".png", "_detected.png")
        cv2.imwrite(output_path, frame)
        print(f"\n✓ Saved annotated image: {output_path}")
    else:
        print("\n⚠ No traffic light bbox detected in image")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_traffic_light_image.py <image_path>")
        print("\nExample:")
        print("  python test_traffic_light_image.py ~/Pictures/traffic_light.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_image(image_path)
