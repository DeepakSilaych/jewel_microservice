import openai
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import base64
import io
import cv2
import mediapipe as mp
import numpy as np
import os
import glob

def crop_jewelry_image(image_path):
    """Crop jewelry image to remove excess whitespace without background removal"""
    img = Image.open(image_path).convert("RGBA")
    
    # Get the bounding box of non-transparent content
    bbox = img.getbbox()
    if bbox:
        # Crop to the bounding box with some padding
        padding = 10
        left, top, right, bottom = bbox
        width, height = img.size
        
        # Add padding while staying within image bounds
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(width, right + padding)
        bottom = min(height, bottom + padding)
        
        img = img.crop((left, top, right, bottom))
        print(f"Cropped jewelry from {width}x{height} to {img.width}x{img.height}")
    else:
        print("No content found to crop, keeping original size")
    
    output_path = image_path.replace('.', '_cropped.')
    img.save(output_path)
    print(f"Jewelry cropped and saved to: {output_path}")
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
    """Use MediaPipe to determine precise pixel coordinates for jewelry placement"""
    
    # Initialize MediaPipe solutions
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    
    # Read image
    image = cv2.imread(model_image_path)
    if image is None:
        print(f"Error: Could not load image {model_image_path}")
        return None, None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    print(f"MediaPipe analysis for {jewelry_type} on {width}x{height} image")
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh, \
    mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands, \
    mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5
    ) as pose:
        
        # Process image with MediaPipe
        face_results = face_mesh.process(image_rgb)
        hand_results = hands.process(image_rgb)
        pose_results = pose.process(image_rgb)
        
        # Store all landmark data for visualization
        landmarks_data = {
            'face_results': face_results,
            'hand_results': hand_results,
            'pose_results': pose_results,
            'jewelry_position': None
        }
        
        jewelry_position = None
        
        if jewelry_type == "earrings":
            # Use Pose landmarks for better ear detection (especially in profile shots)
            if pose_results.pose_landmarks:
                pose_landmarks = pose_results.pose_landmarks.landmark
                left_ear = pose_landmarks[7]   # LEFT_EAR
                right_ear = pose_landmarks[8]  # RIGHT_EAR
                
                left_ear_pos = (int(left_ear.x * width), int(left_ear.y * height))
                right_ear_pos = (int(right_ear.x * width), int(right_ear.y * height))
                
                # Choose ear based on visibility confidence
                if left_ear.visibility > right_ear.visibility and left_ear.visibility > 0.5:
                    jewelry_position = left_ear_pos
                    print(f"Left ear selected (visibility: {left_ear.visibility:.2f})")
                elif right_ear.visibility > 0.5:
                    jewelry_position = right_ear_pos
                    print(f"Right ear selected (visibility: {right_ear.visibility:.2f})")
                
                # Refine position to ear lobe area (slightly below ear landmark)
                if jewelry_position:
                    earring_offset = int(0.015 * height)  # Offset to ear lobe
                    jewelry_position = (jewelry_position[0], jewelry_position[1] + earring_offset)
                    print(f"Earring position detected (pose): {jewelry_position}")
            
            # Fallback to face landmarks if pose detection fails
            elif face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                # Using face mesh ear landmarks as fallback
                left_ear = face_landmarks.landmark[172]  # Left ear area
                right_ear = face_landmarks.landmark[397]  # Right ear area
                
                left_ear_pos = (int(left_ear.x * width), int(left_ear.y * height))
                right_ear_pos = (int(right_ear.x * width), int(right_ear.y * height))
                
                # Use the ear that's more visible (based on z-depth)
                jewelry_position = left_ear_pos if left_ear.z > right_ear.z else right_ear_pos
                print(f"Earring position detected (face fallback): {jewelry_position}")
        
        elif jewelry_type == "necklaces":
            # Use pose landmarks for better neck/collar area detection
            if pose_results.pose_landmarks:
                pose_landmarks = pose_results.pose_landmarks.landmark
                # Use shoulder landmarks to determine necklace position
                left_shoulder = pose_landmarks[11]   # LEFT_SHOULDER
                right_shoulder = pose_landmarks[12]  # RIGHT_SHOULDER
                nose = pose_landmarks[0]             # NOSE (for face center reference)
                
                # Calculate center point between shoulders
                shoulder_center_x = int((left_shoulder.x + right_shoulder.x) * width / 2)
                shoulder_center_y = int((left_shoulder.y + right_shoulder.y) * height / 2)
                
                # Position necklace slightly above shoulder center
                necklace_offset = int(0.08 * height)  # Offset above shoulders
                jewelry_position = (shoulder_center_x, shoulder_center_y - necklace_offset)
                print(f"Necklace position detected (pose): {jewelry_position}")
            
            # Fallback to face landmarks if pose detection fails
            elif face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                # Neck/collar area - use chin and lower face landmarks
                chin = face_landmarks.landmark[175]  # Chin area
                neck_x = int(chin.x * width)
                neck_y = int(chin.y * height) + int(height * 0.1)  # Slightly below chin
                jewelry_position = (neck_x, neck_y)
                print(f"Necklace position detected (face fallback): {jewelry_position}")
        
        elif jewelry_type == "bracelets":
            # Use pose landmarks for better wrist detection
            if pose_results.pose_landmarks:
                pose_landmarks = pose_results.pose_landmarks.landmark
                left_wrist = pose_landmarks[15]   # LEFT_WRIST
                right_wrist = pose_landmarks[16]  # RIGHT_WRIST
                
                left_wrist_pos = (int(left_wrist.x * width), int(left_wrist.y * height))
                right_wrist_pos = (int(right_wrist.x * width), int(right_wrist.y * height))
                
                # Choose wrist based on visibility
                if left_wrist.visibility > right_wrist.visibility and left_wrist.visibility > 0.5:
                    jewelry_position = left_wrist_pos
                    print(f"Left wrist selected (visibility: {left_wrist.visibility:.2f})")
                elif right_wrist.visibility > 0.5:
                    jewelry_position = right_wrist_pos
                    print(f"Right wrist selected (visibility: {right_wrist.visibility:.2f})")
                
                if jewelry_position:
                    print(f"Bracelet position detected (pose): {jewelry_position}")
            
            # Fallback to hand landmarks if pose detection fails
            elif hand_results.multi_hand_landmarks:
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                # Wrist area - use wrist landmark
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
            visualize_landmarks_on_image(model_image_path, landmarks_data, jewelry_type)
        
        return jewelry_position, landmarks_data
    
    print(f"MediaPipe could not detect landmarks for {jewelry_type}")
    return None, None

def get_jewelry_placement_position(model_image_path, jewelry_type):
    """Use MediaPipe to determine precise pixel coordinates for jewelry placement with fallback"""
    
    # Try MediaPipe first
    try:
        position, landmarks_data = get_jewelry_placement_position_mediapipe(model_image_path, jewelry_type)
        if position:
            print(f"MediaPipe positioning successful: {position}")
            return position
    except Exception as e:
        print(f"MediaPipe positioning failed: {e}")
    
    # Fallback to basic positioning
    print("Using fallback positioning")
    model_img = Image.open(model_image_path)
    fallback_positions = {
        "necklaces": (model_img.width // 2, model_img.height // 4),
        "earrings": (model_img.width // 2, model_img.height // 5),
        "bracelets": (model_img.width // 2, model_img.height // 2),
        "rings": (model_img.width // 2, model_img.height // 2)
    }
    fallback_pos = fallback_positions.get(jewelry_type, fallback_positions["necklaces"])
    print(f"Using fallback position: {fallback_pos}")
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
    
    # Step 8: Get MediaPipe-determined placement position (this also saves landmark visualization)
    center_position = get_jewelry_placement_position(model_image_path, jewelry_type)
    
    # Step 9: Convert center position to top-left corner for pasting
    position = (
        center_position[0] - new_jewelry_width // 2,
        center_position[1] - new_jewelry_height // 2
    )
    print(f"9. MediaPipe-determined overlay position: {position} (center: {center_position})")
    
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

def visualize_landmarks_on_image(image_path, landmarks_data, jewelry_type):
    """Create visualization showing detected landmarks on the image"""
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Initialize MediaPipe drawing utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    
    # Create a copy for drawing
    annotated_image = image_rgb.copy()
    
    # Draw face landmarks if available
    if landmarks_data.get('face_results') and landmarks_data['face_results'].multi_face_landmarks:
        for face_landmarks in landmarks_data['face_results'].multi_face_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
    
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
    
    # Draw pose landmarks if available
    if landmarks_data.get('pose_results') and landmarks_data['pose_results'].pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            landmarks_data['pose_results'].pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
        )
    
    # Highlight specific jewelry placement point
    if landmarks_data.get('jewelry_position'):
        x, y = landmarks_data['jewelry_position']
        cv2.circle(annotated_image, (x, y), 15, (255, 255, 0), -1)  # Yellow filled circle
        cv2.circle(annotated_image, (x, y), 20, (0, 0, 0), 2)       # Black border
        
        # Add text label
        label = f"{jewelry_type.upper()} POSITION"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        
        # Position text above the circle
        text_x = x - text_size[0] // 2
        text_y = y - 30
        
        # Draw text background
        cv2.rectangle(annotated_image, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (255, 255, 255), -1)
        
        # Draw text
        cv2.putText(annotated_image, label, (text_x, text_y), 
                   font, font_scale, (0, 0, 0), font_thickness)
    
    # Convert back to PIL Image
    pil_image = Image.fromarray(annotated_image)
    
    # Save with sequential numbering
    filepath = save_numbered_image(pil_image, f"landmarks_{jewelry_type}")
    
    return filepath
