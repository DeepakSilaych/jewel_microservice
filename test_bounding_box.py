#!/usr/bin/env python3
"""
Step 1: Test bounding box detection for jewelry
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def test_jewelry_bounding_box(image_path):
    """Test different methods to detect jewelry bounding box"""
    
    print(f"🔍 TESTING BOUNDING BOX DETECTION: {image_path}")
    print("=" * 50)
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    original_size = img.size
    
    print(f"Original image size: {original_size}")
    
    # Method 1: Simple threshold-based detection
    print("\n1️⃣ METHOD 1: Threshold-based detection")
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Try different threshold values
    for threshold in [50, 100, 150, 200]:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (likely the jewelry)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            area = w * h
            
            print(f"   Threshold {threshold}: Bounding box ({x}, {y}, {w}, {h}), Area: {area}")
        else:
            print(f"   Threshold {threshold}: No contours found")
    
    # Method 2: Edge detection
    print("\n2️⃣ METHOD 2: Edge-based detection")
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate to connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(f"   Edge detection: Bounding box ({x}, {y}, {w}, {h})")
    else:
        print("   Edge detection: No contours found")
    
    # Method 3: Color-based detection (non-black pixels)
    print("\n3️⃣ METHOD 3: Color-based detection")
    
    # Create mask for non-black pixels
    black_threshold = 30
    mask = np.any(img_array > black_threshold, axis=2)
    
    # Find coordinates of non-black pixels
    coords = np.column_stack(np.where(mask))
    
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        print(f"   Color-based: Bounding box ({x_min}, {y_min}, {bbox_width}, {bbox_height})")
    else:
        print("   Color-based: No non-black pixels found")
    
    # Method 4: Histogram-based approach
    print("\n4️⃣ METHOD 4: Histogram analysis")
    
    # Analyze row and column histograms
    row_sums = np.sum(gray, axis=1)
    col_sums = np.sum(gray, axis=0)
    
    # Find rows and columns with significant content
    row_threshold = np.mean(row_sums) * 0.5
    col_threshold = np.mean(col_sums) * 0.5
    
    content_rows = np.where(row_sums > row_threshold)[0]
    content_cols = np.where(col_sums > col_threshold)[0]
    
    if len(content_rows) > 0 and len(content_cols) > 0:
        y_min, y_max = content_rows[0], content_rows[-1]
        x_min, x_max = content_cols[0], content_cols[-1]
        
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        print(f"   Histogram-based: Bounding box ({x_min}, {y_min}, {bbox_width}, {bbox_height})")
    else:
        print("   Histogram-based: No content detected")
    
    print("\n✅ BOUNDING BOX DETECTION TEST COMPLETE")

if __name__ == "__main__":
    import os
    
    if os.path.exists("clipearring.png"):
        test_jewelry_bounding_box("clipearring.png")
    else:
        print("❌ clipearring.png not found")
