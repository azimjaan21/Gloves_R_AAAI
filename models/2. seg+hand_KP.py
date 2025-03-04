import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load models
seg_model = YOLO("runs/segment/train/weights/best.pt")  # Glove Segmentation
pose_model = YOLO("yolo11m-pose.pt")  # Pose Estimation (Only for Hands)

# Choose input type (image or video)
input_source = "chemical.jpg"  # Change to "video.mp4" for video processing
is_video = input_source.endswith((".mp4", ".avi", ".mov"))

# Open video capture if input is a video
if is_video:
    cap = cv2.VideoCapture(input_source)
else:
    cap = cv2.VideoCapture(0)  # Use webcam if needed (change to `cv2.VideoCapture(0)`)

while True:
    # Read frame
    ret, frame = cap.read() if is_video else (True, cv2.imread(input_source))
    if not ret:
        break

    # Run inference
    glove_detections = seg_model.predict(frame, task="segment", conf=0.5)
    pose_detections = pose_model.predict(frame, task="pose", conf=0.5)

    # Extract wrist keypoints
    wrist_keypoints = []
    for result in pose_detections:
        for person in result.keypoints.data:
            left_wrist = person[9][:2] if not torch.isnan(person[9][:2]).any() else None
            right_wrist = person[10][:2] if not torch.isnan(person[10][:2]).any() else None

            if left_wrist is not None:
                left_wrist = tuple(map(int, left_wrist.cpu().numpy()))
            if right_wrist is not None:
                right_wrist = tuple(map(int, right_wrist.cpu().numpy()))

            wrist_keypoints.append((left_wrist, right_wrist))

    # Overlay for transparent mask area
    mask_overlay = frame.copy()

    # Draw glove masks
    for result in glove_detections:
        for i, mask in enumerate(result.masks.xy):
            mask_poly = np.array(mask, dtype=np.int32)
            glove_conf = result.boxes.conf[i].cpu().numpy()

            # Determine if glove has a matching wrist keypoint
            glove_center = np.mean(mask_poly, axis=0).astype(int)
            is_matched = any(
                np.linalg.norm(np.array(wrist) - glove_center) < 50  # Distance threshold
                for wrist_pair in wrist_keypoints
                for wrist in wrist_pair if wrist is not None
            )

            # Set colors
            if is_matched:
                color = (0, 255, 0)  # Green for matched gloves
            else:
                color = (0, 255, 255)  # Yellow for unmatched gloves (ground/missing wrist)

            cv2.fillPoly(mask_overlay, [mask_poly], color=color)  # Mask fill
            cv2.polylines(frame, [mask_poly], isClosed=True, color=color, thickness=2)  # Mask outline

            # Show class label & confidence
            text = f"Gloves {glove_conf:.2f}"
            cv2.putText(frame, text, (glove_center[0], glove_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

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

    if is_video:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break
    else:
        cv2.waitKey(0)
        break

cap.release()
cv2.destroyAllWindows()
