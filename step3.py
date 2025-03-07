import cv2
import torch
import numpy as np
from ultralytics import YOLO
import sys
import os
import torch.nn.functional as F

# Add TCN directory to Python path
tcn_path = os.path.abspath("C:/Users/dalab/Desktop/azimjaan21/Gloves_R_AAAI/TCN/TCN")  
sys.path.append(tcn_path)

from tcn import TemporalConvNet  # Import TCN

print("✅ All libraries imported successfully!")

# Step 2: Load YOLO Models (Segmentation & Pose)
seg_model = YOLO("runs/segment/train/weights/best.pt")  # Glove Segmentation
pose_model = YOLO("yolo11m-pose.pt")  # Wrist Keypoints
print("✅ YOLO models loaded successfully!")

# Step 3: Load Pre-Trained Temporal CNN (TCN)
num_channels = [2, 64, 64, 128, 128]  # TCN Layers (2 input channels: Gloves + Wrist)
tcn_model = TemporalConvNet(num_inputs=2, num_channels=num_channels).cuda().eval()
print("✅ TCN Model loaded successfully!")

# Constants
FRAME_BUFFER_SIZE = 7  # Temporal Window Size
FIXED_MASK_SIZE = (64, 64)  # Standard size for glove masks
MAX_GLOVES = 3  # Maximum number of gloves per frame (adjustable)
DISTANCE_THRESHOLD = 15  # Wrist-Keypoint to Glove Association

# Initialize Temporal Buffer
temporal_buffer = []  # Stores last 7 frames

# Step 4: Start Video Capture
cap = cv2.VideoCapture("tekpe.mp4")
if not cap.isOpened():
    print("❌ ERROR: Video file could not be opened!")
    sys.exit()

print("✅ Video file opened successfully!")

frame_count = 0

# Function to preprocess glove masks to ensure consistent shape
def preprocess_glove_masks(glove_masks, max_gloves=MAX_GLOVES):
    processed_masks = []
    
    if len(glove_masks) > 0:
        glove_masks = sorted(glove_masks, key=lambda x: -x.shape[0])[:max_gloves]

        for mask in glove_masks:
            mask_img = np.zeros(FIXED_MASK_SIZE, dtype=np.uint8)
            cv2.fillPoly(mask_img, [mask], 1)  # Fill detected glove region
            processed_masks.append(mask_img)

    while len(processed_masks) < max_gloves:
        processed_masks.append(np.zeros(FIXED_MASK_SIZE, dtype=np.uint8))

    processed_masks = np.array(processed_masks, dtype=np.float32)
    return torch.tensor(processed_masks).unsqueeze(1)  # Shape: (max_gloves, 1, H, W)

# Function to normalize wrist keypoints
def normalize_keypoints(keypoints, num_keypoints=2):
    if not isinstance(keypoints, list):
        keypoints = []

    fixed_keypoints = []
    for kp in keypoints:
        if kp is None or not isinstance(kp, (list, tuple, torch.Tensor)) or len(kp) != 2:
            fixed_keypoints.append((0.0, 0.0))
        else:
            fixed_keypoints.append((float(kp[0]), float(kp[1])))  # Ensure float conversion

    while len(fixed_keypoints) < num_keypoints:
        fixed_keypoints.append((0.0, 0.0))

    keypoints_array = np.array(fixed_keypoints, dtype=np.float32)
    return torch.tensor(keypoints_array, dtype=torch.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: No frame received from video. Exiting loop.")
        break

    frame_count += 1
    print(f"✅ Processing Frame {frame_count}...")

    mask_overlay = frame.copy()

    # Step 5: Run YOLO-Seg for Glove Detection
    glove_detections = seg_model.predict(frame, conf=0.25)
    print(f"✅ Glove detection completed! Found {len(glove_detections)} results.")

    # Step 6: Run YOLO-Pose for Wrist Keypoints
    pose_detections = pose_model.predict(frame, conf=0.25)
    print(f"✅ Pose detection completed! Found {len(pose_detections)} results.")

    # Step 7: Extract Wrist Keypoints with Safe Indexing
    wrist_keypoints = []
    for result in pose_detections:
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            for person in result.keypoints.data:
                if len(person) > 10:
                    left_wrist = person[9][:2] if not torch.isnan(person[9][:2]).any() else None
                    right_wrist = person[10][:2] if not torch.isnan(person[10][:2]).any() else None
                    wrist_keypoints.append(left_wrist)
                    wrist_keypoints.append(right_wrist)
                else:
                    print("❌ Warning: No wrist keypoints detected in this frame.")
        else:
            print("❌ Warning: No person detected in this frame.")

    if not wrist_keypoints:
        wrist_keypoints = [(0.0, 0.0), (0.0, 0.0)] 

    wrist_keypoints = normalize_keypoints(wrist_keypoints)
    print(f"✅ Extracted {len(wrist_keypoints)} wrist keypoints.")

    # Step 8: Extract Glove Masks
    glove_masks = []
    for result in glove_detections:
        if result.masks is not None and len(result.masks.xy) > 0:
            for mask in result.masks.xy:
                glove_masks.append(np.array(mask, dtype=np.int32))
    print(f"✅ Extracted {len(glove_masks)} glove masks.")

    # Step 9: Store Data in Temporal Buffer
    temporal_buffer.append((glove_masks, wrist_keypoints))
    if len(temporal_buffer) > FRAME_BUFFER_SIZE:
        temporal_buffer.pop(0)
    print(f"✅ Updated temporal buffer. Buffer size: {len(temporal_buffer)}")

    # Step 10: Draw Keypoints & Glove Masks on Frame
    for wrist in wrist_keypoints:
        if isinstance(wrist, torch.Tensor):
            wrist = wrist.cpu().numpy()
        if wrist is not None and wrist[0] > 0 and wrist[1] > 0:
            cv2.circle(mask_overlay, tuple(wrist.astype(int)), 6, (0, 0, 255), -1)

    for mask in glove_masks:
        cv2.fillPoly(mask_overlay, [mask], color=(0, 255, 0))
        cv2.polylines(mask_overlay, [mask], isClosed=True, color=(0, 255, 0), thickness=2)

    frame = cv2.addWeighted(mask_overlay, 0.5, frame, 0.5, 0)

    # Step 11: Display Results
    cv2.namedWindow("Glove Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Glove Detection", 800, 600)
    cv2.imshow("Glove Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("❌ User pressed 'q', exiting script.")
        break
    elif key == ord('p'):
        print("⏸ Paused. Press any key to resume...")
        cv2.waitKey(0)

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Script finished successfully!")
