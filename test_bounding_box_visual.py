#!/usr/bin/env python3
"""
Step 1b: Visual test of bounding box detection
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

def create_visual_bounding_box_test(image_path):
    """Create visual test showing detected bounding boxes"""
    
    print(f"🎨 VISUAL BOUNDING BOX TEST: {image_path}")
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    
    # Method: Edge detection (best from previous test)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        print(f"✅ Detected bounding box: ({x}, {y}, {w}, {h})")
        
        # Create visualization
        img_with_box = img.copy()
        draw = ImageDraw.Draw(img_with_box)
        
        # Draw bounding box
        draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
        
        # Save visualization
        vis_path = "bounding_box_visualization.png"
        img_with_box.save(vis_path)
        print(f"📁 Visualization saved: {vis_path}")
        
        # Test crop
        cropped = img.crop((x, y, x+w, y+h))
        crop_path = "test_cropped_jewelry.png"
        cropped.save(crop_path)
        print(f"📁 Test crop saved: {crop_path}")
        print(f"📏 Cropped size: {cropped.size}")
        
        return x, y, w, h
    else:
        print("❌ No contours found")
        return None

if __name__ == "__main__":
    if os.path.exists("clipearring.png"):
        bbox = create_visual_bounding_box_test("clipearring.png")
        if bbox:
            x, y, w, h = bbox
            print(f"\n📊 BOUNDING BOX RESULTS:")
            print(f"   Position: ({x}, {y})")
            print(f"   Size: {w} x {h}")
            print(f"   Area: {w * h} pixels")
    else:
        print("❌ clipearring.png not found")
