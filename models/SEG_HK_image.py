import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load models
seg_model = YOLO("runs/segment/train/weights/best.pt")  # Glove Segmentation
pose_model = YOLO("yolo11m-pose.pt")  # Pose Estimation (Only for Hands)

# Load image
image_path = "output.png"  # Replace with your image
frame = cv2.imread(image_path)

# Run inference
glove_detections = seg_model.predict(frame, task="segment", conf=0.5)
pose_detections = pose_model.predict(frame, task="pose", conf=0.5)

# Extract wrist keypoints
wrist_keypoints = []
for result in pose_detections:
    for person in result.keypoints.data:
        left_wrist = person[9][:2] if not torch.isnan(person[9][:2]).any() else None
        right_wrist = person[10][:2] if not torch.isnan(person[10][:2]).any() else None
        
        # Convert to integer coordinates
        if left_wrist is not None:
            left_wrist = tuple(map(int, left_wrist.cpu().numpy()))
        if right_wrist is not None:
            right_wrist = tuple(map(int, right_wrist.cpu().numpy()))
        
        wrist_keypoints.append((left_wrist, right_wrist))

# Extract glove masks
glove_masks = []
for result in glove_detections:
    for mask in result.masks.xy:
        glove_masks.append(np.array(mask, dtype=np.int32))

# Overlay for transparent mask area
mask_overlay = frame.copy()

# Draw glove masks
for glove in glove_masks:
    cv2.fillPoly(mask_overlay, [glove], color=(0, 255, 0))  # Green mask fill
    cv2.polylines(frame, [glove], isClosed=True, color=(0, 255, 0), thickness=2)  # Green outline

# Blend overlay (50% opacity)
frame = cv2.addWeighted(mask_overlay, 0.5, frame, 0.5, 0)

# Draw wrist keypoints on top
for (lwrist, rwrist) in wrist_keypoints:
    if lwrist is not None:
        cv2.circle(frame, lwrist, 6, (0, 0, 255), -1)  # Red for left wrist
    if rwrist is not None:
        cv2.circle(frame, rwrist, 6, (0, 0, 255), -1)  # Red for right wrist

# Show output
cv2.imshow("Glove Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
