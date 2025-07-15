import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def detect_face_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        results = face_mesh.process(image_rgb)
        
        landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    landmarks.append((idx, x, y))
        
        return landmarks, image_rgb

def detect_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        results = hands.process(image_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x = int(landmark.x * image.shape[1])
                    y = int(landmark.y * image.shape[0])
                    landmarks.append((idx, x, y))
        
        return landmarks, image_rgb

def detect_pose_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        
        results = pose.process(image_rgb)
        
        landmarks = []
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                landmarks.append((idx, x, y))
        
        return landmarks, image_rgb

def get_jewelry_placement_points(landmarks, jewelry_type):
    if not landmarks:
        return None
    
    placement_points = {}
    
    if jewelry_type == "earrings":
        ear_points = []
        for idx, x, y in landmarks:
            if idx in [234, 454]:  # Left and right ear points
                ear_points.append((x, y))
        placement_points["ears"] = ear_points
    
    elif jewelry_type == "necklaces":
        neck_points = []
        for idx, x, y in landmarks:
            if idx in [10, 151, 9, 175]:  # Neck and throat area
                neck_points.append((x, y))
        placement_points["neck"] = neck_points
    
    elif jewelry_type == "rings":
        finger_tips = []
        for idx, x, y in landmarks:
            if idx in [4, 8, 12, 16, 20]:  # Finger tips
                finger_tips.append((x, y))
        placement_points["fingers"] = finger_tips
    
    elif jewelry_type == "bracelets":
        wrist_points = []
        for idx, x, y in landmarks:
            if idx in [0]:  # Wrist point
                wrist_points.append((x, y))
        placement_points["wrists"] = wrist_points
    
    return placement_points

def visualize_landmarks(image, landmarks, jewelry_type):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    if landmarks:
        x_coords = [x for _, x, y in landmarks]
        y_coords = [y for _, x, y in landmarks]
        plt.scatter(x_coords, y_coords, c='red', s=1)
        
        placement_points = get_jewelry_placement_points(landmarks, jewelry_type)
        if placement_points:
            for location, points in placement_points.items():
                if points:
                    px = [p[0] for p in points]
                    py = [p[1] for p in points]
                    plt.scatter(px, py, c='blue', s=50, label=f'{location} placement')
    
    plt.title(f'{jewelry_type.title()} Landmark Detection')
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'landmarks_{jewelry_type}.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_mediapipe_detection(image_path, jewelry_type):
    print(f"Testing MediaPipe detection for {jewelry_type} on {image_path}")
    
    if jewelry_type == "earrings" or jewelry_type == "necklaces":
        landmarks, image = detect_face_landmarks(image_path)
        print(f"Found {len(landmarks)} face landmarks")
    elif jewelry_type == "rings" or jewelry_type == "bracelets":
        landmarks, image = detect_hand_landmarks(image_path)
        print(f"Found {len(landmarks)} hand landmarks")
    else:
        landmarks, image = detect_pose_landmarks(image_path)
        print(f"Found {len(landmarks)} pose landmarks")
    
    if landmarks:
        placement_points = get_jewelry_placement_points(landmarks, jewelry_type)
        print(f"Jewelry placement points: {placement_points}")
        
        visualize_landmarks(image, landmarks, jewelry_type)
        return placement_points
    else:
        print("No landmarks detected")
        return None

if __name__ == "__main__":
    test_image = "model_earrings_base.webp"
    jewelry_type = "earrings"
    
    placement_points = test_mediapipe_detection(test_image, jewelry_type)
    
    if placement_points:
        print(f"Success! Found placement points for {jewelry_type}: {placement_points}")
    else:
        print(f"Failed to detect landmarks for {jewelry_type}")
