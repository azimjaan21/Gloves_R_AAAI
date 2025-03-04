import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load models
seg_model = YOLO("runs/segment/train/weights/best.pt")  # Glove Segmentation
pose_model = YOLO("yolo11m-pose.pt")  # Pose Estimation (Only for Hands)

# Choose input type (image or video)
input_source = "tekpe.mp4"  # Change to "image.jpg" or "video.mp4"
is_video = input_source.endswith((".mp4", ".avi", ".mov"))

# Open video capture if input is a video
cap = cv2.VideoCapture(input_source if is_video else 0)

while cap.isOpened():
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    glove_detections = seg_model.predict(frame, task="segment", conf=0.25)
    pose_detections = pose_model.predict(frame, task="pose", conf=0.25)

    # Extract wrist keypoints
    wrist_keypoints = []
    for result in pose_detections:
        if result.keypoints is not None and len(result.keypoints.data) > 0:  # ✅ Check if keypoints exist
            for person in result.keypoints.data:
                if len(person) > 10:  # ✅ Ensure wrist keypoints exist before accessing them
                    left_wrist = person[9][:2] if not torch.isnan(person[9][:2]).any() else None
                    right_wrist = person[10][:2] if not torch.isnan(person[10][:2]).any() else None

                    if left_wrist is not None:
                        left_wrist = tuple(map(int, left_wrist.cpu().numpy()))
                    if right_wrist is not None:
                        right_wrist = tuple(map(int, right_wrist.cpu().numpy()))

                    wrist_keypoints.append((left_wrist, right_wrist))

    # Overlay for transparent mask area
    mask_overlay = frame.copy()

    # Process glove detections (ONLY DRAW MATCHED GLOVES)
    for result in glove_detections:
        if result.masks is not None and len(result.masks.xy) > 0:  # ✅ Check if masks exist
            for i, mask in enumerate(result.masks.xy):
                mask_poly = np.array(mask, dtype=np.int32)
                glove_conf = result.boxes.conf[i].cpu().numpy() if result.boxes is not None else 0.0

                # Determine if glove has a matching wrist keypoint
                glove_center = np.mean(mask_poly, axis=0).astype(int)
                is_matched = any(
                    np.linalg.norm(np.array(wrist) - glove_center) < 50  # Distance threshold
                    for wrist_pair in wrist_keypoints
                    for wrist in wrist_pair if wrist is not None
                )

                if is_matched:  # ✅ Only draw matched gloves (false alarms removed!)
                    cv2.fillPoly(mask_overlay, [mask_poly], color=(0, 255, 0))  # Green for matched gloves
                    cv2.polylines(frame, [mask_poly], isClosed=True, color=(0, 255, 0), thickness=2)

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

    # Press 'q' to exit video processing
    if is_video:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKey(0)
        break

cap.release()
cv2.destroyAllWindows()
