from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv11-Pose model
pose_model = YOLO("yolo11m-pose.pt")  # Update with your pose model path

# Define wrist keypoint indexes (from YOLO-Pose keypoint order)
WRIST_KEYPOINTS = [9, 10]  # Right wrist (9), Left wrist (10)

# Open webcam
cap = cv2.VideoCapture(0)  # Change to 1 if using an external webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame. Exiting...")
        break

    # Run YOLOv11-Pose on the frame
    pose_results = pose_model.predict(source=frame, task="pose", device="cuda", conf=0.25, save=False)

    # Extract wrist keypoints
    wrist_keypoints = []
    for result in pose_results:
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            for kp in result.keypoints.xy:
                if len(kp) > max(WRIST_KEYPOINTS):  # Ensure keypoints exist
                    right_wrist_x, right_wrist_y = kp[9]  # Right wrist
                    left_wrist_x, left_wrist_y = kp[10]  # Left wrist

                    wrist_keypoints.append((int(right_wrist_x), int(right_wrist_y)))  # Right wrist
                    wrist_keypoints.append((int(left_wrist_x), int(left_wrist_y)))  # Left wrist

    # Draw wrist keypoints on the frame
    for wrist_x, wrist_y in wrist_keypoints:
        cv2.circle(frame, (wrist_x, wrist_y), 5, (0, 255, 0), -1)  # Green for wrist keypoints

    # Display the result
    cv2.imshow("YOLOv11-Pose Wrist Keypoints", frame)

    # Press 'q' to exit the webcam stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
