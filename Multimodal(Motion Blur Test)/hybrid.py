import os
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Load Models
seg_model = YOLO("runs/segment/train/weights/gloves.pt")  # Glove Segmentation
pose_model = YOLO("yolo11m-pose.pt")  # Pose Estimation (For Wrist Keypoints)

# Video Path & Output Directory
video_path = "gloves.mp4"  # Input video file path
output_dir = "Multimodal(Motion Blur Test)/hybrid_video_results"  # Folder for processed results
frames_output_dir = os.path.join(output_dir, "frames")  # Subfolder for output frames
csv_path = os.path.join(output_dir, "hybrid_evaluation_video.csv")

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(frames_output_dir, exist_ok=True)

# Expected gloves per frame (Assuming 2 gloves per frame)
EXPECTED_GLOVES = 2  
DISTANCE_THRESHOLD = 50  # Pixels for wrist-glove matching

# Open video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ ERROR: Unable to open video file!")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = os.path.join(output_dir, "output_hybrid.mp4")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Store evaluation results
evaluation_data = []

frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if no more frames

    frame_index += 1
    overlay = frame.copy()  # Copy for visualization

    # Run inference using YOLO models
    seg_results = seg_model.predict(frame, task="segment", conf=0.25, save=False)
    pose_results = pose_model.predict(frame, task="pose", conf=0.25, save=False)

    # Extract Wrist Keypoints
    wrist_keypoints = []
    for result in pose_results:
        if result.keypoints is not None:
            for kp in result.keypoints.xy:
                if len(kp) > 10:  # Ensure keypoints exist
                    left_wrist = tuple(map(int, kp[9][:2].cpu().numpy())) if not torch.isnan(kp[9][:2]).any() else None
                    right_wrist = tuple(map(int, kp[10][:2].cpu().numpy())) if not torch.isnan(kp[10][:2]).any() else None
                    if left_wrist is not None and len(left_wrist) == 2:
                        wrist_keypoints.append(left_wrist)
                    if right_wrist is not None and len(right_wrist) == 2:
                        wrist_keypoints.append(right_wrist)

    # Extract Predicted Gloves & No-Gloves Masks
    glove_masks = []
    no_glove_masks = []
    for result in seg_results:
        if result.masks is not None:
            for mask, cls in zip(result.masks.xy, result.boxes.cls):
                mask_pts = mask.astype(np.int32)
                if int(cls) == 0:
                    glove_masks.append(mask_pts)  # Class 0: Gloves
                else:
                    no_glove_masks.append(mask_pts)  # Class 1: No Gloves

    # ✅ Fix: Prevent iteration error if wrist keypoints are empty
    valid_glove_masks = []
    valid_no_glove_masks = []

    if wrist_keypoints:
        for mask in glove_masks:
            for wrist in wrist_keypoints:
                try:
                    min_distance = min(
                        np.linalg.norm(np.array(wrist) - np.array(mask_pt))
                        for mask_pt in mask
                    )
                    if min_distance < DISTANCE_THRESHOLD:
                        valid_glove_masks.append(mask)
                        break  # Stop checking after finding a match
                except:
                    pass  # Avoid crashing due to bad mask formatting

        for mask in no_glove_masks:
            for wrist in wrist_keypoints:
                try:
                    min_distance = min(
                        np.linalg.norm(np.array(wrist) - np.array(mask_pt))
                        for mask_pt in mask
                    )
                    if min_distance < DISTANCE_THRESHOLD:
                        valid_no_glove_masks.append(mask)
                        break
                except:
                    pass  # Avoid crashing due to bad mask formatting

    # Draw Valid Gloves (Green) and No-Gloves (Red)
    for mask in valid_glove_masks:
        cv2.fillPoly(overlay, [mask], (0, 255, 0))  # Green for gloves
    for mask in valid_no_glove_masks:
        cv2.fillPoly(overlay, [mask], (0, 0, 255))  # Red for no gloves

    # Draw Wrist Keypoints in Blue
    for wrist_x, wrist_y in wrist_keypoints:
        cv2.circle(overlay, (wrist_x, wrist_y), 6, (255, 0, 0), -1)

    # Blend overlay with transparency
    output_image = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    # Write to video
    out.write(output_image)

    # Save processed frame as an image
    output_frame_path = os.path.join(frames_output_dir, f"frame_{frame_index:04d}.jpg")
    cv2.imwrite(output_frame_path, output_image)

    # Compute consistency accuracy
    detected_gloves = len(valid_glove_masks)
    consistency_acc = (detected_gloves / EXPECTED_GLOVES) * 100  # In percentage

    # Save evaluation results
    evaluation_data.append({
        "Frame": frame_index,
        "Detected Gloves": detected_gloves,
        "Expected Gloves": EXPECTED_GLOVES,
        "Consistency Accuracy (%)": round(consistency_acc, 2)
    })

    print(f"✅ Processed Frame {frame_index}/{total_frames} → Saved to {output_frame_path}")

# Save evaluation results to CSV
df = pd.DataFrame(evaluation_data)
df.to_csv(csv_path, index=False)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Evaluation results saved: {csv_path}")
print(f"✅ Processed video saved: {output_video_path}")
print(f"✅ Processed frames saved in: {frames_output_dir}")
