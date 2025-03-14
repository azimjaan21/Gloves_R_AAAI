from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load pre-trained models
pose_model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\yolo11m-pose.pt")  # Wrist detection
segmentation_model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\runs\segment\train\weights\best.pt")  # Glove segmentation

# Define dataset path
dataset_path = r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\final\train\images"  # Update path
output_folder = r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\final\train_results_wrist_gloves"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Define wrist keypoint indexes (from YOLO-Pose keypoint order)
WRIST_KEYPOINTS = [9, 10]  # Right wrist (9), Left wrist (10)

# Process each image in the dataset
for img_file in os.listdir(dataset_path):
    if img_file.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(dataset_path, img_file)
        image = cv2.imread(img_path)

        # Run YOLO-Pose for wrist keypoints
        pose_results = pose_model.predict(source=img_path, task="pose", device="cuda", conf=0.25, save=False)

        # Run YOLO-Seg for glove segmentation
        seg_results = segmentation_model.predict(source=img_path, task="segment", device="cuda", conf=0.25, save=False)

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

        # Extract glove segmentation masks
        glove_masks = []
        for result in seg_results:
            if result.masks is not None:
                for mask in result.masks.xy:
                    glove_masks.append(mask.astype(np.int32))  # Ensure correct OpenCV format

        # Perform keypoint-based filtering: Ensure glove mask contains a wrist keypoint
        valid_glove_masks = []
        for mask_pts in glove_masks:
            mask_valid = any(cv2.pointPolygonTest(mask_pts, wrist, False) >= 0 for wrist in wrist_keypoints)

            if mask_valid:
                valid_glove_masks.append(mask_pts)

        # Draw wrist keypoints on the image
        for wrist_x, wrist_y in wrist_keypoints:
            cv2.circle(image, (wrist_x, wrist_y), 5, (0, 255, 0), -1)  # Green for wrist keypoints

        # Draw glove masks before filtering (blue) and after filtering (red)
        for mask_pts in glove_masks:
            cv2.polylines(image, [mask_pts], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue: All detected gloves
        for mask_pts in valid_glove_masks:
            cv2.polylines(image, [mask_pts], isClosed=True, color=(0, 0, 255), thickness=2)  # Red: Valid gloves on hand

        # Save processed image in the same output folder
        output_img_path = os.path.join(output_folder, img_file)  # Save with the original filename
        cv2.imwrite(output_img_path, image)
        print(f"✅ Processed: {img_file} → Saved to {output_folder}")

print(f"✅ Wrist-glove verification complete! All results saved in: {output_folder}")
