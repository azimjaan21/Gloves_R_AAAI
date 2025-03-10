import cv2
import torch
import numpy as np
import os
import sys
import torch.nn.functional as F
import pandas as pd
from ultralytics import YOLO


# Add TCN directory to Python path
tcn_path = os.path.abspath("C:/Users/dalab/Desktop/azimjaan21/Gloves_R_AAAI/TCN/TCN")  
sys.path.append(tcn_path)

from tcn import TemporalConvNet



print("✅ All libraries imported successfully!")

# Step 1: Load YOLO Models (Segmentation & Pose)
seg_model = YOLO("runs/segment/train/weights/best.pt")  # Glove Segmentation
pose_model = YOLO("yolo11m-pose.pt")  # Wrist Keypoints
print("✅ YOLO models loaded successfully!")

# Step 2: Load Pre-Trained Temporal CNN (TCN)
num_channels = [2, 64, 64, 128, 128]  # TCN Layers (2 input channels: Gloves + Wrist)
tcn_model = TemporalConvNet(num_inputs=2, num_channels=num_channels).cuda().eval()
print("✅ TCN Model loaded successfully!")

# Constants
FRAME_BUFFER_SIZE = 7  # Temporal Window Size
DISTANCE_THRESHOLD = 30  # Maximum wrist-to-glove distance (in pixels) for a match
FIXED_MASK_SIZE = (64, 64)  # Standard size for glove masks
MAX_GLOVES = 3  # Maximum gloves per frame

# Initialize Temporal Buffer
temporal_buffer = []  # Stores last 7 frames

# Step 3: Start Video Capture
cap = cv2.VideoCapture("gloves.mp4")
if not cap.isOpened():
    print("❌ ERROR: Video file could not be opened!")
    sys.exit()

print("✅ Video file opened successfully!")

frame_count = 0

# Initialize metrics tracking
metrics_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: No frame received from video. Exiting loop.")
        break

    frame_count += 1
    print(f"✅ Processing Frame {frame_count}...")

    mask_overlay = frame.copy()

    # Step 4: Run YOLO-Seg for Glove Detection
    glove_detections = seg_model.predict(frame, conf=0.25)
    print(f"✅ Glove detection completed! Found {len(glove_detections)} results.")

    # Step 5: Run YOLO-Pose for Wrist Keypoints
    pose_detections = pose_model.predict(frame, conf=0.25)
    print(f"✅ Pose detection completed! Found {len(pose_detections)} results.")

    # Step 6: Extract Wrist Keypoints
    wrist_keypoints = []
    for result in pose_detections:
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            for person in result.keypoints.data:
                if len(person) > 10:
                    left_wrist = person[9][:2] if not torch.isnan(person[9][:2]).any() else None
                    right_wrist = person[10][:2] if not torch.isnan(person[10][:2]).any() else None
                    wrist_keypoints.append(left_wrist)
                    wrist_keypoints.append(right_wrist)
    
    # Convert wrist keypoints to numpy for easy processing
    wrist_keypoints = [w.cpu().numpy() if isinstance(w, torch.Tensor) else w for w in wrist_keypoints]
    print(f"✅ Extracted {len(wrist_keypoints)} wrist keypoints.")

    # Step 7: Extract & Match Glove Masks
    matched_gloves = []
    
    for result in glove_detections:
        if result.masks is not None and len(result.masks.xy) > 0:
            for mask in result.masks.xy:
                mask_poly = np.array(mask, dtype=np.int32)

                # Find the closest wrist keypoint to this glove mask
                is_matched = any(
                    min(np.linalg.norm(np.array(wrist) - np.array(mask_pt)) for mask_pt in mask_poly) < DISTANCE_THRESHOLD
                    for wrist in wrist_keypoints if wrist is not None
                )

                if is_matched:
                    matched_gloves.append(mask_poly)  # ✅ Only matched gloves are stored

    print(f"✅ Matched {len(matched_gloves)} gloves with wrist keypoints.")

    # Step 8: Store Data in Temporal Buffer
    temporal_buffer.append((matched_gloves, wrist_keypoints))
    if len(temporal_buffer) > FRAME_BUFFER_SIZE:
        temporal_buffer.pop(0)
    print(f"✅ Updated temporal buffer. Buffer size: {len(temporal_buffer)}")

    # Step 9: Evaluate Performance
    true_positive = len(matched_gloves)
    false_positive = len(glove_detections) - true_positive
    false_negative = max(0, len(wrist_keypoints) - true_positive)

    # Calculate Precision, Recall, F1-score
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Save per-frame results
    metrics_data.append({
        "Frame": frame_count,
        "True Positives": true_positive,
        "False Positives": false_positive,
        "False Negatives": false_negative,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score
    })

    # Step 10: Draw Wrist Keypoints & Matched Gloves
    for wrist in wrist_keypoints:
        if wrist is not None and wrist[0] > 0 and wrist[1] > 0:
            cv2.circle(mask_overlay, tuple(wrist.astype(int)), 6, (0, 0, 255), -1)  # 🔴 Red keypoints

    for mask in matched_gloves:
        cv2.fillPoly(mask_overlay, [mask], color=(0, 255, 0))  # 🟢 Green overlay for gloves
        cv2.polylines(mask_overlay, [mask], isClosed=True, color=(0, 255, 0), thickness=2)  # Green bounding box

    final_output = cv2.addWeighted(mask_overlay, 0.5, frame, 0.5, 0)

    # Step 11: Display Results
    cv2.imshow("Glove Detection", final_output)

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

# Step 12: Save Evaluation Results to CSV
df = pd.DataFrame(metrics_data)
df.to_csv("multimodal_evaluation.csv", index=False)
print("✅ Evaluation results saved as 'multimodal_evaluation.csv'")
