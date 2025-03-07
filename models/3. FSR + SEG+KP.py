print("✅ Script started successfully!")

import warnings
warnings.simplefilter("ignore", FutureWarning)  # Suppress weight_norm warnings

import cv2
import torch
import numpy as np
from ultralytics import YOLO

print("✅ YOLO and necessary libraries imported!")

from tcn import TemporalConvNet  # Ensure TCN is imported correctly
print("✅ TCN Model imported successfully!")

# Load YOLO models
seg_model = YOLO("runs/segment/train/weights/best.pt")
pose_model = YOLO("yolo11m-pose.pt")

print("✅ YOLO models loaded successfully!")

# Load Temporal CNN (TCN)
num_channels = [2, 64, 64, 128, 128]
tcn_model = TemporalConvNet(num_inputs=2, num_channels=num_channels).cuda().eval()

print("✅ TCN Model loaded successfully!")

# Start Video Processing
cap = cv2.VideoCapture("video.mp4")
if not cap.isOpened():
    print("❌ ERROR: Video file could not be opened!")
else:
    print("✅ Video file opened successfully!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: No frame received from video. Exiting loop.")
        break

    print("✅ Frame received, processing started...")

    # Run YOLO-Seg
    glove_detections = seg_model.predict(frame, conf=0.25)
    print("✅ Glove detection completed!")

    # Run YOLO-Pose
    pose_detections = pose_model.predict(frame, conf=0.25)
    print("✅ Pose detection completed!")

    # Show Frame (Debugging)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❌ User pressed 'q', exiting script.")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Script finished successfully!")
