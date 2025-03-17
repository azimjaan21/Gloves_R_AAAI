import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load models
seg_model = YOLO("runs/segment/train/weights/gloves.pt")  # Glove Segmentation
pose_model = YOLO("yolo11m-pose.pt")  # Pose Estimation (Only for Hands)

# Choose input type (image or video)
input_source = "gloves.mp4"  # Change to "image.jpg" or "video.mp4"
is_video = input_source.endswith((".mp4", ".avi", ".mov"))

# Open video capture if input is a video
cap = cv2.VideoCapture(input_source if is_video else 0)

# Get video properties for saving output
if is_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_video(hy).mp4", fourcc, fps, (width, height))

# Wrist keypoint indexes in YOLOv11-Pose
WRIST_INDEXES = [9, 10]  # Left wrist = 9, Right wrist = 10
DISTANCE_THRESHOLD = 100  # Pixels for wrist-glove matching

# Function to compute distance from wrist to closest glove polygon side
def point_to_line_distance(point, line_start, line_end):
    """Computes the shortest Euclidean distance between a point and a line segment."""
    line = np.array(line_end) - np.array(line_start)
    if np.dot(line, line) == 0:
        return np.linalg.norm(point - line_start)  # If same points, return point distance

    t = max(0, min(1, np.dot(point - line_start, line) / np.dot(line, line)))
    projection = line_start + t * line  # Projection on line segment
    return np.linalg.norm(point - projection)

while cap.isOpened():
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Create an overlay for semi-transparent masks
    mask_overlay = frame.copy()

    # Run inference
    glove_detections = seg_model.predict(frame, conf=0.25)
    pose_detections = pose_model.predict(frame, conf=0.25)

    # Extract wrist keypoints
    wrist_keypoints = []
    for result in pose_detections:
        if result.keypoints is not None and len(result.keypoints.data) > 0:  # ✅ Check if keypoints exist
            for person in result.keypoints.data:
                if len(person) > max(WRIST_INDEXES):  # ✅ Ensure wrist keypoints exist before accessing them
                    left_wrist = person[9][:2] if not torch.isnan(person[9][:2]).any() else None
                    right_wrist = person[10][:2] if not torch.isnan(person[10][:2]).any() else None

                    if left_wrist is not None:
                        left_wrist = tuple(map(int, left_wrist.cpu().numpy()))
                    if right_wrist is not None:
                        right_wrist = tuple(map(int, right_wrist.cpu().numpy()))

                    wrist_keypoints.append((left_wrist, right_wrist))

    # Process glove detections (Apply wrist-side filtering)
    for result in glove_detections:
        if result.masks is not None and len(result.masks.xy) > 0:  # ✅ Check if masks exist
            for i, mask in enumerate(result.masks.xy):
                mask_poly = np.array(mask, dtype=np.int32)
                glove_conf = result.boxes.conf[i].cpu().numpy() if result.boxes is not None else 0.0

                # ✅ Find the closest side of the glove to the wrist
                is_matched = any(
                    min(point_to_line_distance(np.array(wrist), mask_poly[j], mask_poly[(j + 1) % len(mask_poly)])
                        for j in range(len(mask_poly))) < DISTANCE_THRESHOLD
                    for wrist_pair in wrist_keypoints
                    for wrist in wrist_pair if wrist is not None
                )

                # ✅ Only show matched gloves (Filtered gloves are NOT displayed)
                if is_matched:
                    cv2.fillPoly(mask_overlay, [mask_poly], color=(0, 255, 0))  # ✅ Green overlay fill
                    cv2.polylines(frame, [mask_poly], isClosed=True, color=(0, 255, 0), thickness=2)  # ✅ Green outline

                    # Show class label & confidence
                    text = f"Gloves {glove_conf:.2f}"
                    cv2.putText(frame, text, (mask_poly[0][0], mask_poly[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Blend overlay with 50% transparency
    frame = cv2.addWeighted(mask_overlay, 0.5, frame, 0.5, 0)

    # Draw wrist keypoints (Now in **RED**)
    for (lwrist, rwrist) in wrist_keypoints:
        if lwrist is not None:
            cv2.circle(frame, lwrist, 6, (0, 0, 255), -1)  # 🔴 Red for left wrist
        if rwrist is not None:
            cv2.circle(frame, rwrist, 6, (0, 0, 255), -1)  # 🔴 Red for right wrist

    # Write the processed frame to the output video
    if is_video:
        out.write(frame)

    # Show output
    cv2.imshow("Glove Detection", frame)

    # Press 'q' to exit video processing
    if is_video:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKey(0)
        break

# Release video capture & writer
cap.release()
if is_video:
    out.release()
cv2.destroyAllWindows()
print("✅ Processed video saved as 'out_video.mp4'")
