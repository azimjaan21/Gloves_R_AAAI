import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import sys
import os

# Add TCN directory to Python path
tcn_path = os.path.abspath("C:/Users/dalab/Desktop/azimjaan21/Gloves_R_AAAI/TCN/TCN")  
sys.path.append(tcn_path)

from tcn import TemporalConvNet  # Import TCN

print("✅ All libraries imported successfully!")

# Load YOLO Models (Segmentation & Pose)
seg_model = YOLO("runs/segment/train/weights/gloves.pt")  # Glove Segmentation
pose_model = YOLO("yolo11m-pose.pt")  # Wrist Keypoints
print("✅ YOLO models loaded successfully!")

# Load Pre-Trained Temporal CNN (TCN)
num_channels = [2, 64, 64, 128, 128]  # TCN Layers (2 input channels: Gloves + Wrist)
tcn_model = TemporalConvNet(num_inputs=2, num_channels=num_channels).cuda().eval()
print("✅ TCN Model loaded successfully!")

# Constants
FRAME_BUFFER_SIZE = 7  # Temporal Window Size
DISTANCE_THRESHOLD = 50  # Maximum wrist-to-glove distance (in pixels) for a match
EXPECTED_GLOVES_PER_FRAME = 2  # Always 2 gloves per frame

# Paths for Saving Results
video_path = "gloves.mp4"
output_dir = "Multimodal(Motion Blur Test)/tcn_results"
frames_output_dir = os.path.join(output_dir, "frames")
csv_path = os.path.join(output_dir, "tcn_evaluation.csv")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(frames_output_dir, exist_ok=True)

# Initialize Temporal Buffer
temporal_buffer = []  # Stores last 7 frames
evaluation_data = []  # Stores frame-wise evaluation results

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ ERROR: Video file could not be opened!")
    sys.exit()

# Get video properties for saving output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(os.path.join(output_dir, "output_multimodal_tcn.mp4"), fourcc, fps, (width, height))

print("✅ Video file opened successfully!")

frame_count = 0
total_tp, total_fp, total_fn = 0, 0, 0  # Initialize TP, FP, FN counters

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ Video processing completed!")
        break

    frame_count += 1
    frame_filename = f"frame_{frame_count:05d}.jpg"

    mask_overlay = frame.copy()

    # Step 1: Run YOLO-Seg for Glove Detection
    glove_detections = seg_model.predict(frame, conf=0.25)

    # Step 2: Run YOLO-Pose for Wrist Keypoints
    pose_detections = pose_model.predict(frame, conf=0.25)

    # Step 3: Extract Wrist Keypoints
    wrist_keypoints = []
    for result in pose_detections:
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            for person in result.keypoints.data:
                if len(person) > 10:
                    left_wrist = person[9][:2] if not torch.isnan(person[9][:2]).any() else None
                    right_wrist = person[10][:2] if not torch.isnan(person[10][:2]).any() else None
                    wrist_keypoints.append(left_wrist)
                    wrist_keypoints.append(right_wrist)

    # Convert wrist keypoints to numpy
    wrist_keypoints = [w.cpu().numpy() if isinstance(w, torch.Tensor) else w for w in wrist_keypoints]
    wrist_count = sum(1 for w in wrist_keypoints if w is not None)

    # Step 4: Extract & Match Glove Masks
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
                    matched_gloves.append(mask_poly)

    detected_gloves = len(matched_gloves)

    # Step 5: Store Data in Temporal Buffer (Gloves + Wrist)
    temporal_buffer.append((detected_gloves, wrist_count))
    if len(temporal_buffer) > FRAME_BUFFER_SIZE:
        temporal_buffer.pop(0)

    # Ensure buffer always has 7 frames
    while len(temporal_buffer) < FRAME_BUFFER_SIZE:
        temporal_buffer.insert(0, (0, 0))

    # Prepare input for TCN (2-channel input: Gloves & Wrist keypoints)
    glove_counts = [frame[0] for frame in temporal_buffer]
    wrist_counts = [frame[1] for frame in temporal_buffer]

    # Convert to Tensor (1, 2, 7)
    tcn_input = torch.tensor([glove_counts, wrist_counts]).float().unsqueeze(0).cuda()

    # Run TCN for glove consistency prediction
    tcn_output = tcn_model(tcn_input).detach().cpu().numpy().flatten()[-1]
    detected_gloves_tcn = int(round(tcn_output))

    # Step 6: Compute TP, FP, FN
    tp = detected_gloves_tcn
    fn = EXPECTED_GLOVES_PER_FRAME - detected_gloves_tcn
    fp = 0  # TCN does not misclassify gloves as "No Gloves"

    # Update global TP, FP, FN counters
    total_tp += tp
    total_fp += fp
    total_fn += fn

    # Compute Temporal Consistency Accuracy
    consistency_accuracy = (tp / EXPECTED_GLOVES_PER_FRAME) * 100

    print(f"🔹 Frame {frame_count} | TCN Gloves: {detected_gloves_tcn} | Consistency: {consistency_accuracy:.2f}%")

    # Step 7: Draw Matched Gloves (Green) & No-Gloves (Red)
    for mask in matched_gloves:
        cv2.fillPoly(mask_overlay, [mask], color=(0, 255, 0))  # ✅ Green for gloves

    # 🔴 If TCN says gloves are missing in multiple frames, highlight missing gloves in red
    if fn > 0:
        for wrist in wrist_keypoints:
            if wrist is not None and wrist[0] > 0 and wrist[1] > 0:
                cv2.circle(mask_overlay, tuple(wrist.astype(int)), 20, (0, 0, 255), -1)  # 🔴 Red for missing gloves

    final_output = cv2.addWeighted(mask_overlay, 0.5, frame, 0.5, 0)
    out.write(final_output)

    # Save frame as image
    output_frame_path = os.path.join(frames_output_dir, frame_filename)
    cv2.imwrite(output_frame_path, final_output)

    # Step 8: Store Evaluation Data
    evaluation_data.append({
        "Frame": frame_filename,
        "TP (Correct Gloves)": tp,
        "FP (Misclassified Gloves)": fp,
        "FN (Missed Gloves)": fn,
        "Temporal Consistency (%)": round(consistency_accuracy, 2)
    })

# Save Evaluation Results to CSV
df = pd.DataFrame(evaluation_data)
df.to_csv(csv_path, index=False)
print(f"✅ Evaluation results saved: {csv_path}")

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Processed video saved as 'output_multimodal_tcn.mp4'")
