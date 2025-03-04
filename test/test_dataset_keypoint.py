from ultralytics import YOLO
import cv2
import os

# Load pre-trained YOLOv11-Pose model
pose_model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\yolo11m-pose.pt")  # Update with your model path

# Define dataset path
dataset_path = r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\final\valid\images"  # Update path
output_folder = r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\results_wrist_detection"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Define wrist keypoint indexes (from YOLO-Pose keypoint order)
WRIST_KEYPOINTS = [9, 10]  # Right wrist (9), Left wrist (10)

# Process each image in the dataset
for img_file in os.listdir(dataset_path):
    if img_file.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(dataset_path, img_file)
        image = cv2.imread(img_path)

        # Run YOLOv11-Pose
        pose_results = pose_model.predict(source=img_path, task="pose", device="cuda", conf=0.25, save=False)

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

        # Draw wrist keypoints on the image
        for wrist_x, wrist_y in wrist_keypoints:
            cv2.circle(image, (wrist_x, wrist_y), 5, (0, 255, 0), -1)  # Green for wrist keypoints

        # Save processed image in the same output folder
        output_img_path = os.path.join(output_folder, img_file)  # Save with the original filename
        cv2.imwrite(output_img_path, image)
        print(f"✅ Processed: {img_file} → Saved to {output_folder}")

print(f"✅ Wrist keypoint detection complete! All results saved in: {output_folder}")
