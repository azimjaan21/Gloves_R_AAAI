import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

# Load trained YOLO-Seg model
model = YOLO(r'runs/segment/train/weights/best.pt')

# Define input source (video or image)
input_source = "gloves.mp4"
is_video = input_source.endswith((".mp4", ".avi", ".mov"))

# Open video file
cap = cv2.VideoCapture(input_source) if is_video else None

# Define mask color
mask_color = (0, 255, 0)  # Green in BGR

# Initialize lists for evaluation
metrics_data = []  # Stores per-frame evaluation results
frame_count = 0

while is_video and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # Run inference
    results = model.predict(frame, conf=0.25)

    # Process detections
    true_positive = 0
    false_positive = 0
    false_negative = 0  # Placeholder (requires GT masks for accurate FN count)
    
    for result in results:
        masks = result.masks
        if masks is not None:
            for i, mask in enumerate(masks.xy):
                mask = np.array(mask, dtype=np.int32)

                # Confidence & Label
                conf = result.boxes.conf[i].item()
                true_positive += 1  # Approximating all detections as TP (no GT available)
                
    # Approximate false positives (Assumption: Some detections might be FP)
    false_positive = max(0, len(results) - true_positive)
    
    # Compute Precision, Recall, F1-score
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Store results
    metrics_data.append({
        "Frame": frame_count,
        "True Positives": true_positive,
        "False Positives": false_positive,
        "False Negatives": false_negative,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_score
    })

# Convert results to DataFrame
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv("baseline_yolo_seg_evaluation.csv", index=False)
print("✅ Evaluation results saved as 'baseline_yolo_seg_evaluation.csv'")

# Cleanup
cap.release() if is_video else None
cv2.destroyAllWindows()