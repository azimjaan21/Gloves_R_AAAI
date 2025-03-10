import cv2
import os

# Set input video path
video_path = "gloves.mp4"  # Change this to your video file path
output_folder = "frames"  # Change this to your desired output folder
frame_rate = 1  # Extract every frame (set to higher values to skip frames)

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video
    
    # Save frame based on the specified frame rate
    if frame_count % frame_rate == 0:
        frame_filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
    
    frame_count += 1

cap.release()
print(f"Extraction complete! {saved_count} frames saved in {output_folder}")
