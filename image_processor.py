import openai
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import base64
import io
import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import insightface
from insightface.app import FaceAnalysis

def crop_jewelry_image(image_path):
    """Simple and reliable jewelry cropping using edge detection"""
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    
    print(f"🔍 Cropping jewelry image: {original_size}")
    
    # Convert to numpy for OpenCV processing
    img_array = np.array(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect jewelry parts
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour (jewelry)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(original_size[0] - x, w + 2 * padding)
        h = min(original_size[1] - y, h + 2 * padding)
        
        # Crop the image
        cropped_img = img.crop((x, y, x + w, y + h))
        
        print(f"✅ Edge detection crop: {original_size} → {cropped_img.size}")
        print(f"   Bounding box: ({x}, {y}, {w}, {h})")
        
    else:
        print("⚠️ No edges detected, using original image")
        cropped_img = img
    
    # Save using sequential numbering system
    jewelry_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = save_numbered_image(cropped_img, f"cropped_{jewelry_name}")
    print(f"📁 Cropped jewelry saved: {output_path}")
    return output_path

def get_image_height_mm(model_image_path, jewelry_type):
    with open(model_image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    prompt = """You are a professional anthropometric analyst and computer vision expert specializing in human body proportion measurement. Your task is to perform precise dimensional analysis of this model image.

ANALYSIS METHODOLOGY:
1. Identify the visible human anatomical features in the image
2. Reference standard anthropometric data: average human height = 1700mm
3. Use anatomical landmarks and proportional relationships to calculate measurements
4. Apply scientific estimation based on established human body ratios

TASK: Estimate the total height of the visible area in this image in millimeters, considering the human proportional relationships and the 1700mm reference height.

Return only the numerical value in millimeters."""
    
    response = openai.chat.completions.create(
        model="o3",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                ]
            }
        ],
    )
    
    try:
        height_str = response.choices[0].message.content.strip()
        height_mm = float(''.join(filter(str.isdigit, height_str.split('.')[0])))
        print(f"AI height analysis result: {height_mm}mm")
        return height_mm
    except:
        fallback_heights = {
            "necklaces": 300,
            "earrings": 200,
            "bracelets": 150,
            "rings": 100
        }
        fallback_height = fallback_heights.get(jewelry_type, 200)
        print(f"AI analysis failed, using fallback: {fallback_height}mm")
        return fallback_height

def calculate_jewelry_size_mm(jewelry_size_str):
    print(f"Parsing jewelry size: '{jewelry_size_str}'")
    
    if not jewelry_size_str:
        print("No size provided, using default: 20mm")
        return 20
    
    size_lower = jewelry_size_str.lower()
    
    if 'cm' in size_lower:
        try:
            numbers = [float(s) for s in size_lower.replace('x', ' ').split() if s.replace('.', '').isdigit()]
            result = max(numbers) * 10
            print(f"Converted from cm: {max(numbers)}cm = {result}mm")
            return result
        except:
            print("CM parsing failed, using default: 20mm")
            return 20
    elif 'mm' in size_lower:
        try:
            numbers = [float(s) for s in size_lower.replace('x', ' ').split() if s.replace('.', '').isdigit()]
            result = max(numbers)
            print(f"MM value extracted: {result}mm")
            return result
        except:
            print("MM parsing failed, using default: 20mm")
            return 20
    else:
        print("No unit found, using default: 20mm")
        return 20

def get_jewelry_placement_position_mediapipe(model_image_path, jewelry_type):
    """Use BlazePose (MediaPipe Pose) to determine precise pixel coordinates for jewelry placement"""
    
    # Initialize MediaPipe Pose (BlazePose)
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Read image
    image = cv2.imread(model_image_path)
    if image is None:
        print(f"Error: Could not load image {model_image_path}")
        return None, None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    print(f"BlazePose analysis for {jewelry_type} on {width}x{height} image")
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,  # Higher complexity for better accuracy
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose, \
    mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        
        # Process image with MediaPipe
        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)
        
        # Store all landmark data for visualization
        landmarks_data = {
            'pose_results': pose_results,
            'hand_results': hand_results,
            'jewelry_position': None
        }
        
        jewelry_position = None
        
        if jewelry_type == "earrings":
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                # BlazePose landmark reference:
                # 0: Nose, 1: Left Eye Inner, 2: Left Eye, 3: Left Eye Outer
                # 4: Right Eye Inner, 5: Right Eye, 6: Right Eye Outer
                # 7: Left Ear, 8: Right Ear
                
                left_ear = landmarks[7]   # Left ear
                right_ear = landmarks[8]  # Right ear
                
                # Check visibility and choose the better ear
                if left_ear.visibility > right_ear.visibility and left_ear.visibility > 0.5:
                    # Use left ear
                    ear_x = int(left_ear.x * width)
                    ear_y = int(left_ear.y * height)
                    print(f"Left ear selected (visibility: {left_ear.visibility:.3f})")
                elif right_ear.visibility > 0.5:
                    # Use right ear
                    ear_x = int(right_ear.x * width)
                    ear_y = int(right_ear.y * height)
                    print(f"Right ear selected (visibility: {right_ear.visibility:.3f})")
                else:
                    # Fallback: estimate using eye and face geometry
                    left_eye = landmarks[2]   # Left eye
                    right_eye = landmarks[5]  # Right eye
                    nose = landmarks[0]       # Nose
                    
                    # Choose side based on profile orientation
                    if left_eye.visibility > right_eye.visibility:
                        # Left profile - use left side
                        ear_x = int((left_eye.x * width) - 0.05 * width)  # Slightly behind eye
                        ear_y = int((left_eye.y + nose.y) * height / 2)
                        print("Estimated left ear position from face geometry")
                    else:
                        # Right profile - use right side
                        ear_x = int((right_eye.x * width) + 0.05 * width)  # Slightly behind eye
                        ear_y = int((right_eye.y + nose.y) * height / 2)
                        print("Estimated right ear position from face geometry")
                
                # Apply earring offset (lower position for ear lobe)
                if 'ear_x' in locals():
                    earring_offset = int(0.02 * height)  # Offset to ear lobe position
                    jewelry_position = (ear_x, ear_y + earring_offset)
                    print(f"Earring position detected (BlazePose): {jewelry_position}")
        
        elif jewelry_type == "necklaces":
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                # Use shoulder landmarks for necklace positioning
                left_shoulder = landmarks[11]   # Left shoulder
                right_shoulder = landmarks[12]  # Right shoulder
                
                # Calculate center point between shoulders
                shoulder_center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
                shoulder_center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)
                
                # Position necklace above shoulder center (collar area)
                necklace_offset = int(0.08 * height)
                jewelry_position = (shoulder_center_x, shoulder_center_y - necklace_offset)
                print(f"Necklace position detected (BlazePose): {jewelry_position}")
        
        elif jewelry_type == "bracelets":
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                left_wrist = landmarks[15]   # Left wrist
                right_wrist = landmarks[16]  # Right wrist
                
                # Choose wrist based on visibility
                if left_wrist.visibility > right_wrist.visibility and left_wrist.visibility > 0.5:
                    jewelry_position = (int(left_wrist.x * width), int(left_wrist.y * height))
                    print(f"Left wrist selected (visibility: {left_wrist.visibility:.3f})")
                elif right_wrist.visibility > 0.5:
                    jewelry_position = (int(right_wrist.x * width), int(right_wrist.y * height))
                    print(f"Right wrist selected (visibility: {right_wrist.visibility:.3f})")
                
                if jewelry_position:
                    print(f"Bracelet position detected (BlazePose): {jewelry_position}")
            
            # Fallback to hand landmarks if pose wrists not visible
            elif hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                wrist = hand_landmarks.landmark[0]  # Wrist landmark
                jewelry_position = (int(wrist.x * width), int(wrist.y * height))
                print(f"Bracelet position detected (hand fallback): {jewelry_position}")
        
        elif jewelry_type == "rings":
            if hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                # Ring finger landmarks for better positioning
                ring_finger_mcp = hand_landmarks.landmark[13]   # Ring finger MCP (base)
                ring_finger_tip = hand_landmarks.landmark[16]   # Ring finger tip
                
                # Position ring at middle of ring finger
                ring_x = int((ring_finger_mcp.x + ring_finger_tip.x) * width / 2)
                ring_y = int((ring_finger_mcp.y + ring_finger_tip.y) * height / 2)
                jewelry_position = (ring_x, ring_y)
                print(f"Ring position detected (hand): {jewelry_position}")
        
        # Store the jewelry position in landmarks data
        landmarks_data['jewelry_position'] = jewelry_position
        
        if jewelry_position:
            # Save landmark visualization
            visualize_blazepose_landmarks(model_image_path, landmarks_data, jewelry_type)
        
        return jewelry_position, landmarks_data
    
    print(f"BlazePose could not detect landmarks for {jewelry_type}")
    return None, None

def get_jewelry_placement_position_insightface(model_image_path, jewelry_type):
    """Use InsightFace 3D face analysis for precise jewelry positioning"""
    
    print(f"InsightFace 3D analysis for {jewelry_type}")
    
    # Load image
    img = cv2.imread(model_image_path)
    if img is None:
        print(f"Error: Could not load image {model_image_path}")
        return None, None
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    try:
        # Initialize InsightFace with 3D landmark support
        app = FaceAnalysis(
            providers=['CPUExecutionProvider'], 
            allowed_modules=['detection', 'landmark_3d']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Detect faces
        faces = app.get(rgb)
        if not faces:
            print("No face detected by InsightFace")
            return None, None
        
        # Use the first (most confident) face
        face = faces[0]
        
        # Get 3D landmarks (68 points)
        if not hasattr(face, 'landmark_3d_68'):
            print("3D landmarks not available")
            return None, None
            
        lm3d = np.array(face.landmark_3d_68)  # shape (68, 3)
        
        # Store face analysis data for visualization
        face_data = {
            'face': face,
            'landmarks_3d': lm3d,
            'bbox': face.bbox,
            'jewelry_position': None
        }
        
        jewelry_position = None
        
        if jewelry_type == "earrings":
            # InsightFace 68-point landmark indices for ear area:
            # 0-16: Jaw line (0=left jaw, 16=right jaw)
            # 2: Lower left jaw near ear
            # 14: Lower right jaw near ear
            
            left_ear_idx = 2   # Left jaw near ear
            right_ear_idx = 14 # Right jaw near ear
            
            left_point = lm3d[left_ear_idx]
            right_point = lm3d[right_ear_idx]
            
            # Use 3D z-depth for visibility (higher z = more visible/closer)
            if left_point[2] > right_point[2]:
                # Left ear more visible
                chosen_point = left_point
                ear_side = "left"
            else:
                # Right ear more visible
                chosen_point = right_point
                ear_side = "right"
            
            # Convert normalized coordinates to pixel coordinates
            x_px = int(chosen_point[0] * width)
            y_px = int(chosen_point[1] * height)
            
            # Apply ear lobe offset (slightly downward for realistic positioning)
            ear_lobe_offset = int(0.015 * height)
            jewelry_position = (x_px, y_px + ear_lobe_offset)
            
            print(f"Earring position: {jewelry_position} ({ear_side} ear, z-depth: {chosen_point[2]:.3f})")
        
        elif jewelry_type == "necklaces":
            # Use jaw line and chin points for necklace positioning
            chin_idx = 8  # Bottom of chin
            left_jaw_idx = 5
            right_jaw_idx = 11
            
            chin_point = lm3d[chin_idx]
            left_jaw = lm3d[left_jaw_idx]
            right_jaw = lm3d[right_jaw_idx]
            
            # Calculate necklace position at collar area
            center_x = int((left_jaw[0] + right_jaw[0]) * width / 2)
            necklace_y = int(chin_point[1] * height) + int(0.12 * height)  # Below chin
            
            jewelry_position = (center_x, necklace_y)
            print(f"Necklace position: {jewelry_position}")
        
        elif jewelry_type == "bracelets":
            # For bracelets, we still need hand detection
            # InsightFace focuses on face, so we'll use a fallback approach
            # Position bracelet at lower part of image (estimated wrist area)
            bracelet_x = width // 2
            bracelet_y = int(height * 0.7)  # Lower portion of image
            jewelry_position = (bracelet_x, bracelet_y)
            print(f"Bracelet position (estimated): {jewelry_position}")
        
        elif jewelry_type == "rings":
            # Similar to bracelets, estimate hand area
            ring_x = int(width * 0.6)  # Slightly right of center
            ring_y = int(height * 0.8)  # Lower portion
            jewelry_position = (ring_x, ring_y)
            print(f"Ring position (estimated): {jewelry_position}")
        
        # Store jewelry position
        face_data['jewelry_position'] = jewelry_position
        
        if jewelry_position:
            # Save visualization
            visualize_insightface_landmarks(model_image_path, face_data, jewelry_type)
        
        return jewelry_position, face_data
        
    except Exception as e:
        print(f"InsightFace analysis failed: {e}")
        return None, None

def get_jewelry_placement_position_haar(model_image_path, jewelry_type):
    """Use Haar cascade ear detection for earring positioning"""
    
    if jewelry_type != "earrings":
        print("Haar cascade only supports earring detection")
        return None, None
    
    try:
        from ear_detection import get_earring_position_haar
        print("Using Haar cascade ear detection...")
        
        position = get_earring_position_haar(model_image_path)
        
        if position:
            # Create simple visualization data
            haar_data = {
                'method': 'haar_cascade',
                'jewelry_position': position
            }
            
            # Save simple visualization
            visualize_haar_detection(model_image_path, haar_data, jewelry_type)
            
            return position, haar_data
        else:
            print("Haar cascade ear detection failed")
            return None, None
            
    except ImportError:
        print("Haar cascade ear detection not available")
        return None, None
    except Exception as e:
        print(f"Haar cascade error: {e}")
        return None, None

def get_jewelry_placement_position(model_image_path, jewelry_type):
    """Main jewelry positioning function - Haar for earrings, MediaPipe for others"""
    
    if jewelry_type == "earrings":
        # Use Haar cascade for earring detection
        print("🎯 Using Haar cascade for earring detection...")
        try:
            position = get_earring_position_haar(model_image_path)
            if position:
                print(f"✅ Haar cascade earring positioning successful: {position}")
                return position
            else:
                print("❌ Haar cascade failed, using MediaPipe fallback...")
        except Exception as e:
            print(f"❌ Haar cascade error: {e}, using MediaPipe fallback...")
    
    # Use MediaPipe for all other jewelry types or as earring fallback
    print("🤖 Using MediaPipe positioning...")
    try:
        position, landmarks_data = get_jewelry_placement_position_mediapipe(model_image_path, jewelry_type)
        if position:
            print(f"✅ MediaPipe positioning successful: {position}")
            return position
        else:
            print("❌ MediaPipe failed, using basic fallback...")
    except Exception as e:
        print(f"❌ MediaPipe error: {e}, using basic fallback...")
    
    # Final fallback to basic positioning
    print("📐 Using basic geometric fallback...")
    model_img = Image.open(model_image_path)
    fallback_positions = {
        "necklaces": (model_img.width // 2, model_img.height // 4),
        "earrings": (model_img.width // 2, model_img.height // 5),
        "bracelets": (model_img.width // 2, model_img.height // 2),
        "rings": (model_img.width // 2, model_img.height // 2)
    }
    fallback_pos = fallback_positions.get(jewelry_type, fallback_positions["necklaces"])
    print(f"✅ Using fallback position: {fallback_pos}")
    return fallback_pos

def resize_and_overlay(model_image_path, jewelry_image_path, jewelry_size_str, jewelry_type):
    print(f"\n=== STARTING JEWELRY PLACEMENT PIPELINE ===")
    print(f"Model: {model_image_path}")
    print(f"Jewelry: {jewelry_image_path}")
    print(f"Type: {jewelry_type}")
    print(f"Size: {jewelry_size_str}")
    
    # Step 1: Save original model image
    model_img = Image.open(model_image_path).convert("RGBA")
    original_model_path = save_numbered_image(model_img, f"original_model_{jewelry_type}")
    print(f"1. Original model saved: {original_model_path}")
    
    # Step 2: Crop jewelry image
    processed_jewelry = crop_jewelry_image(jewelry_image_path)
    jewelry_img = Image.open(processed_jewelry).convert("RGBA")
    cropped_jewelry_path = save_numbered_image(jewelry_img, f"cropped_jewelry_{jewelry_type}")
    print(f"2. Jewelry cropped and saved: {cropped_jewelry_path}")
    
    # Step 3: Estimate image height
    image_height_mm = get_image_height_mm(model_image_path, jewelry_type)
    print(f"3. Image height estimated: {image_height_mm}mm")
    
    # Step 4: Calculate jewelry size
    jewelry_size_mm = calculate_jewelry_size_mm(jewelry_size_str)
    print(f"4. Jewelry size calculated: {jewelry_size_mm}mm")
    
    print(f"5. Images loaded - Model: {model_img.size}, Jewelry: {jewelry_img.size}")
    
    # Step 6: Calculate scaling
    image_height_px = model_img.height
    jewelry_height_px = jewelry_img.height
    
    scale_factor = (jewelry_size_mm / jewelry_height_px) / (image_height_mm / image_height_px)
    print(f"6. Scale factor calculated: {scale_factor}")
    
    new_jewelry_width = int(jewelry_img.width * scale_factor)
    new_jewelry_height = int(jewelry_img.height * scale_factor)
    print(f"7. New jewelry size: {new_jewelry_width}x{new_jewelry_height}px")
    
    # Step 7: Resize jewelry
    jewelry_resized = jewelry_img.resize((new_jewelry_width, new_jewelry_height), Image.Resampling.LANCZOS)
    resized_jewelry_path = save_numbered_image(jewelry_resized, f"resized_jewelry_{jewelry_type}")
    print(f"8. Resized jewelry saved: {resized_jewelry_path}")
    
    # Step 8: Get InsightFace/MediaPipe-determined placement position (this also saves landmark visualization)
    center_position = get_jewelry_placement_position(model_image_path, jewelry_type)
    
    # Step 9: Convert center position to top-left corner for pasting
    position = (
        center_position[0] - new_jewelry_width // 2,
        center_position[1] - new_jewelry_height // 2
    )
    print(f"9. InsightFace/MediaPipe-determined overlay position: {position} (center: {center_position})")
    
    # Step 10: Create final composite image
    final_img = model_img.copy()
    final_img.paste(jewelry_resized, position, jewelry_resized)
    
    # Save final result with sequential numbering
    final_path = save_numbered_image(final_img, f"final_{jewelry_type}_composite")
    print(f"10. Final composite image saved: {final_path}")
    
    print(f"\n=== PIPELINE COMPLETE ===")
    print(f"All images saved in 'img' folder with sequential numbering")
    
    return final_path

def ensure_img_directory():
    """Create img directory if it doesn't exist"""
    img_dir = "img"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"Created directory: {img_dir}")
    return img_dir

def get_next_image_number():
    """Get the next sequential number for image saving"""
    img_dir = ensure_img_directory()
    
    # Find all numbered images in the img directory
    pattern = os.path.join(img_dir, "*.png")
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return 1
    
    # Extract numbers from filenames
    numbers = []
    for file in existing_files:
        basename = os.path.basename(file)
        try:
            # Extract number from filename (assumes format like "1_description.png")
            number = int(basename.split('_')[0])
            numbers.append(number)
        except (ValueError, IndexError):
            continue
    
    if numbers:
        return max(numbers) + 1
    else:
        return 1

def save_numbered_image(image, description, extension=".png"):
    """Save image with sequential numbering in img folder"""
    img_dir = ensure_img_directory()
    number = get_next_image_number()
    
    filename = f"{number}_{description}{extension}"
    filepath = os.path.join(img_dir, filename)
    
    if isinstance(image, str):
        # If image is a path, copy the file
        import shutil
        shutil.copy2(image, filepath)
    else:
        # If image is a PIL Image object, save it
        image.save(filepath)
    
    print(f"Saved: {filepath}")
    return filepath

def visualize_blazepose_landmarks(image_path, landmarks_data, jewelry_type):
    """Create visualization showing BlazePose landmarks and jewelry position"""
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Initialize MediaPipe drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    
    # Create a copy for drawing
    annotated_image = image_rgb.copy()
    
    # Draw pose landmarks if available
    if landmarks_data.get('pose_results') and landmarks_data['pose_results'].pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            landmarks_data['pose_results'].pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
        )
        
        # Highlight specific landmarks for debugging
        pose_landmarks = landmarks_data['pose_results'].pose_landmarks.landmark
        
        if jewelry_type == "earrings":
            # Highlight ear landmarks
            left_ear = pose_landmarks[7]
            right_ear = pose_landmarks[8]
            
            # Draw ear landmarks with visibility info
            left_ear_pos = (int(left_ear.x * width), int(left_ear.y * height))
            right_ear_pos = (int(right_ear.x * width), int(right_ear.y * height))
            
            cv2.circle(annotated_image, left_ear_pos, 8, (255, 0, 0), -1)  # Blue for left ear
            cv2.circle(annotated_image, right_ear_pos, 8, (0, 0, 255), -1)  # Red for right ear
            
            # Add visibility text
            cv2.putText(annotated_image, f"L:{left_ear.visibility:.2f}", 
                       (left_ear_pos[0]-20, left_ear_pos[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_image, f"R:{right_ear.visibility:.2f}", 
                       (right_ear_pos[0]-20, right_ear_pos[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        elif jewelry_type == "necklaces":
            # Highlight shoulder landmarks
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            
            left_shoulder_pos = (int(left_shoulder.x * width), int(left_shoulder.y * height))
            right_shoulder_pos = (int(right_shoulder.x * width), int(right_shoulder.y * height))
            
            cv2.circle(annotated_image, left_shoulder_pos, 8, (255, 0, 0), -1)
            cv2.circle(annotated_image, right_shoulder_pos, 8, (0, 0, 255), -1)
        
        elif jewelry_type == "bracelets":
            # Highlight wrist landmarks
            left_wrist = pose_landmarks[15]
            right_wrist = pose_landmarks[16]
            
            left_wrist_pos = (int(left_wrist.x * width), int(left_wrist.y * height))
            right_wrist_pos = (int(right_wrist.x * width), int(right_wrist.y * height))
            
            cv2.circle(annotated_image, left_wrist_pos, 8, (255, 0, 0), -1)
            cv2.circle(annotated_image, right_wrist_pos, 8, (0, 0, 255), -1)
            
            # Add visibility text
            cv2.putText(annotated_image, f"L:{left_wrist.visibility:.2f}", 
                       (left_wrist_pos[0]-20, left_wrist_pos[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated_image, f"R:{right_wrist.visibility:.2f}", 
                       (right_wrist_pos[0]-20, right_wrist_pos[1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw hand landmarks if available
    if landmarks_data.get('hand_results') and landmarks_data['hand_results'].multi_hand_landmarks:
        for hand_landmarks in landmarks_data['hand_results'].multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            )
    
    # Highlight jewelry placement point
    if landmarks_data.get('jewelry_position'):
        x, y = landmarks_data['jewelry_position']
        
        # Draw large highlight circle
        cv2.circle(annotated_image, (x, y), 20, (0, 255, 255), -1)  # Yellow filled circle
        cv2.circle(annotated_image, (x, y), 25, (0, 0, 0), 3)       # Black border
        
        # Add text label
        label = f"{jewelry_type.upper()} POSITION"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Position text above the circle
        text_x = x - text_size[0] // 2
        text_y = y - 40
        
        # Draw text background
        cv2.rectangle(annotated_image, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (255, 255, 255), -1)
        
        # Draw text
        cv2.putText(annotated_image, label, (text_x, text_y), 
                   font, font_scale, (0, 0, 0), font_thickness)
        
        # Add coordinate info
        coord_text = f"({x}, {y})"
        cv2.putText(annotated_image, coord_text, (text_x, text_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Convert back to PIL Image
    pil_image = Image.fromarray(annotated_image)
    
    # Save with sequential numbering
    filepath = save_numbered_image(pil_image, f"blazepose_landmarks_{jewelry_type}")
    
    return filepath

def visualize_insightface_landmarks(image_path, face_data, jewelry_type):
    """Create visualization showing InsightFace 3D landmarks and jewelry position"""
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    height, width = img.shape[:2]
    
    # Create visualization
    vis_img = img.copy()
    
    if face_data and face_data.get('landmarks_3d') is not None:
        lm3d = face_data['landmarks_3d']
        
        # Draw all 68 3D landmarks
        for i, point in enumerate(lm3d):
            x = int(point[0] * width)
            y = int(point[1] * height)
            z_depth = point[2]
            
            # Color code by depth (closer = brighter)
            intensity = min(255, max(0, int(z_depth * 100 + 128)))
            color = (0, intensity, 255 - intensity)  # Blue to red gradient
            
            cv2.circle(vis_img, (x, y), 2, color, -1)
            
            # Label key landmarks
            if jewelry_type == "earrings" and i in [2, 14]:  # Ear area landmarks
                cv2.circle(vis_img, (x, y), 6, (0, 255, 255), 2)
                cv2.putText(vis_img, f"{i}({z_depth:.2f})", (x-10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            elif jewelry_type == "necklaces" and i in [5, 8, 11]:  # Jaw/chin landmarks
                cv2.circle(vis_img, (x, y), 6, (0, 255, 0), 2)
                cv2.putText(vis_img, f"{i}", (x-10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw face bounding box
        if face_data.get('bbox') is not None:
            bbox = face_data['bbox'].astype(int)
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(vis_img, "InsightFace Detection", (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Highlight jewelry placement point
    if face_data and face_data.get('jewelry_position'):
        x, y = face_data['jewelry_position']
        
        # Draw large highlight circle
        cv2.circle(vis_img, (x, y), 25, (0, 255, 255), -1)  # Yellow filled circle
        cv2.circle(vis_img, (x, y), 30, (0, 0, 0), 3)       # Black border
        
        # Add text label
        label = f"{jewelry_type.upper()} (InsightFace 3D)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Position text above the circle
        text_x = x - text_size[0] // 2
        text_y = y - 50
        
        # Draw text background
        cv2.rectangle(vis_img, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (255, 255, 255), -1)
        
        # Draw text
        cv2.putText(vis_img, label, (text_x, text_y), 
                   font, font_scale, (0, 0, 0), font_thickness)
        
        # Add coordinate info
        coord_text = f"({x}, {y})"
        cv2.putText(vis_img, coord_text, (text_x, text_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add legend
    legend_y = 30
    cv2.putText(vis_img, "InsightFace 3D Landmarks", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_img, "Blue=Far, Red=Close", (10, legend_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Convert to PIL and save
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(vis_img_rgb)
    
    # Save with sequential numbering
    filepath = save_numbered_image(pil_image, f"insightface_3d_{jewelry_type}")
    
    return filepath

def get_earring_position_haar(image_path):
    """Detect ear position using Haar cascades and return bottom of ear for earring placement"""
    
    # Check if cascade files exist
    cascade_dir = "cascade_files"
    left_ear_cascade_path = os.path.join(cascade_dir, 'haarcascade_mcs_leftear.xml')
    right_ear_cascade_path = os.path.join(cascade_dir, 'haarcascade_mcs_rightear.xml')
    
    # Create cascade directory if it doesn't exist
    if not os.path.exists(cascade_dir):
        os.makedirs(cascade_dir)
        print(f"Created directory: {cascade_dir}")
    
    # Check for cascade files
    if not os.path.exists(left_ear_cascade_path) or not os.path.exists(right_ear_cascade_path):
        print("Haar cascade files not found. Please download:")
        print(f"- {left_ear_cascade_path}")
        print(f"- {right_ear_cascade_path}")
        print("From: https://github.com/opencv/opencv/tree/master/data/haarcascades")
        return None
    
    try:
        # Load cascade classifiers
        left_ear_cascade = cv2.CascadeClassifier(left_ear_cascade_path)
        right_ear_cascade = cv2.CascadeClassifier(right_ear_cascade_path)
        
        if left_ear_cascade.empty():
            print("Error: Could not load left ear cascade")
            return None
        
        if right_ear_cascade.empty():
            print("Error: Could not load right ear cascade")
            return None
        
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        
        # Detect left ears
        left_ears = left_ear_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Detect right ears
        right_ears = right_ear_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Choose the best ear detection
        best_ear = None
        best_confidence = 0
        
        # Process left ears
        for (x, y, w, h) in left_ears:
            # Calculate confidence based on size and position
            ear_area = w * h
            confidence = ear_area / (width * height)  # Normalized area
            
            if confidence > best_confidence:
                best_confidence = confidence
                # Position earring at bottom center of detected ear
                ear_center_x = x + w // 2
                ear_bottom_y = y + h  # Bottom of ear for earring placement
                best_ear = (ear_center_x, ear_bottom_y)
                print(f"Left ear detected: {(x, y, w, h)}, confidence: {confidence:.3f}")
        
        # Process right ears
        for (x, y, w, h) in right_ears:
            ear_area = w * h
            confidence = ear_area / (width * height)
            
            if confidence > best_confidence:
                best_confidence = confidence
                ear_center_x = x + w // 2
                ear_bottom_y = y + h  # Bottom of ear for earring placement
                best_ear = (ear_center_x, ear_bottom_y)
                print(f"Right ear detected: {(x, y, w, h)}, confidence: {confidence:.3f}")
        
        if best_ear:
            print(f"Best ear position for earring: {best_ear} (confidence: {best_confidence:.3f})")
            
            # Save visualization
            visualize_haar_detection(image_path, left_ears, right_ears, best_ear)
            
            return best_ear
        else:
            print("No ears detected with Haar cascades")
            return None
            
    except Exception as e:
        print(f"Haar cascade detection error: {e}")
        return None

def visualize_haar_detection(image_path, left_ears, right_ears, chosen_position):
    """Create visualization of Haar cascade ear detection"""
    
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    vis_image = image.copy()
    
    # Draw left ear detections in blue
    for (x, y, w, h) in left_ears:
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(vis_image, "L", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Draw right ear detections in red
    for (x, y, w, h) in right_ears:
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(vis_image, "R", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Highlight chosen earring position
    if chosen_position:
        x, y = chosen_position
        cv2.circle(vis_image, (x, y), 15, (0, 255, 255), -1)  # Yellow circle
        cv2.circle(vis_image, (x, y), 20, (0, 0, 0), 3)       # Black border
        
        # Add label
        cv2.putText(vis_image, "EARRING", (x - 30, y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis_image, f"({x},{y})", (x - 30, y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add title
    cv2.putText(vis_image, "Haar Cascade Ear Detection", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Convert and save
    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(vis_image_rgb)
    
    # Save with sequential numbering
    filepath = save_numbered_image(pil_image, "haar_ear_detection")
    
    return filepath
