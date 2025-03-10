import cv2
import numpy as np
import torch
import pandas as pd
from ultralytics import YOLO

# Load models
seg_model = YOLO("runs/segment/train/weights/best.pt")  # Glove Segmentation
pose_model = YOLO("yolo11m-pose.pt")  # Pose Estimation (Only for Hands)

# Choose input type (image or video)
input_source = "gloves.mp4"  
is_video = input_source.endswith((".mp4", ".avi", ".mov"))

# Open video capture
cap = cv2.VideoCapture(input_source) if is_video else None

# Constants
WRIST_INDEXES = [9, 10]  # Left wrist = 9, Right wrist = 10
DISTANCE_THRESHOLD = 20  # Pixels for wrist-glove matching

# Initialize lists for evaluation
metrics_data = []  # Stores per-frame evaluation results
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"✅ Processing Frame {frame_count}...")

    # Run inference
    glove_detections = seg_model.predict(frame, conf=0.25)
    pose_detections = pose_model.predict(frame, conf=0.25)

    # Extract wrist keypoints
    wrist_keypoints = []
    for result in pose_detections:
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            for person in result.keypoints.data:
                if len(person) > max(WRIST_INDEXES):
                    left_wrist = person[9][:2] if not torch.isnan(person[9][:2]).any() else None
                    right_wrist = person[10][:2] if not torch.isnan(person[10][:2]).any() else None
                    wrist_keypoints.append(left_wrist)
                    wrist_keypoints.append(right_wrist)
    
    wrist_keypoints = [w.cpu().numpy() if isinstance(w, torch.Tensor) else w for w in wrist_keypoints]
    print(f"✅ Extracted {len(wrist_keypoints)} wrist keypoints.")

    # Process glove detections
    matched_gloves = 0
    false_positives = 0
    
    for result in glove_detections:
        if result.masks is not None and len(result.masks.xy) > 0:
            for mask in result.masks.xy:
                mask_poly = np.array(mask, dtype=np.int32)

                # Match glove with wrist keypoints
                is_matched = any(
                    min(np.linalg.norm(np.array(wrist) - np.array(mask_pt)) for mask_pt in mask_poly) < DISTANCE_THRESHOLD
                    for wrist in wrist_keypoints if wrist is not None
                )

                if is_matched:
                    matched_gloves += 1  # ✅ True Positive
                else:
                    false_positives += 1  # ❌ False Positive
    
    false_negatives = max(0, len(wrist_keypoints) - matched_gloves)  # Placeholder

    # Compute Precision, Recall, F1-score
    precision = matched_gloves / (matched_gloves + false_positives) if (matched_gloves + false_positives) > 0 else 0
    recall = matched_gloves / (matched_gloves + false_negatives) if (matched_gloves + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Store results
    metrics_data.append({
        "Frame": frame_count,
        "True Positives": matched_gloves,
        "False Positives": false_positives,
        "False Negatives": false_negatives,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score
    })

# Convert results to DataFrame
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv("yolo_seg_pose_evaluation.csv", index=False)
print("✅ Evaluation results saved as 'yolo_seg_pose_evaluation.csv'")

# Cleanup
cap.release() if is_video else None
cv2.destroyAllWindows()
