from ultralytics import YOLO
import cv2
import os
import numpy as np
import pandas as pd

# Load YOLO models
pose_model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\yolo11m-pose.pt")  # Wrist detection
segmentation_model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\runs\segment\train\weights\best.pt")  # Glove segmentation

# Dataset path
dataset_path = r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\evaluation3\images"
output_folder = r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\hybrid_results"
os.makedirs(output_folder, exist_ok=True)

# CSV log for evaluation metrics
csv_path = os.path.join(output_folder, "evaluation_results.csv")

# Define wrist keypoint indexes (from YOLO-Pose keypoint order)
WRIST_KEYPOINTS = [9, 10]  # Right wrist (9), Left wrist (10)
DISTANCE_THRESHOLD = 25  # Pixels for "near" wrist-glove matching

# Function to compute shortest distance from a point (wrist) to a polygon (glove mask)
def point_to_line_distance(point, line_start, line_end):
    """Computes the shortest Euclidean distance between a point and a line segment."""
    line = np.array(line_end) - np.array(line_start)
    if np.dot(line, line) == 0:
        return np.linalg.norm(point - line_start)  # If same points, return direct distance

    t = max(0, min(1, np.dot(point - line_start, line) / np.dot(line, line)))
    projection = line_start + t * line  # Projection on the segment
    return np.linalg.norm(point - projection)

def is_wrist_near_glove(wrist, mask_pts, threshold=DISTANCE_THRESHOLD):
    """Check if wrist keypoint is within distance threshold of a glove mask."""
    for i in range(len(mask_pts) - 1):  # Iterate over polygon edges
        if point_to_line_distance(wrist, mask_pts[i], mask_pts[i + 1]) < threshold:
            return True
    return False

# Evaluation storage
evaluation_data = []

# Process each image
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

        # Perform keypoint-based filtering: Ensure glove mask contains or is near a wrist keypoint
        valid_glove_masks = []
        for mask_pts in glove_masks:
            mask_valid = any(
                cv2.pointPolygonTest(mask_pts, wrist, False) >= 0 or is_wrist_near_glove(wrist, mask_pts)
                for wrist in wrist_keypoints
            )
            if mask_valid:
                valid_glove_masks.append(mask_pts)

        # Count metrics
        total_gloves_detected = len(glove_masks)
        valid_gloves = len(valid_glove_masks)
        false_positives = total_gloves_detected - valid_gloves  # Gloves removed by keypoint filtering
        false_negatives = 0  # Assume we don't have ground truth, so we can't compute FN directly

        precision = valid_gloves / (valid_gloves + false_positives + 1e-6)  # Avoid division by zero
        recall = valid_gloves / (valid_gloves + false_negatives + 1e-6)

        # Store evaluation data
        evaluation_data.append({
            "Image": img_file,
            "Total Gloves Detected": total_gloves_detected,
            "Valid Gloves After Filtering": valid_gloves,
            "False Positives": false_positives,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4)
        })

        # Draw wrist keypoints on the image
        for wrist_x, wrist_y in wrist_keypoints:
            cv2.circle(image, (wrist_x, wrist_y), 5, (0, 255, 0), -1)  # Green for wrist keypoints

        # Draw glove masks before filtering (blue) and after filtering (red)
        for mask_pts in glove_masks:
            cv2.polylines(image, [mask_pts], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue: All detected gloves
        for mask_pts in valid_glove_masks:
            cv2.polylines(image, [mask_pts], isClosed=True, color=(0, 0, 255), thickness=2)  # Red: Valid gloves on hand

        # Save processed image
        output_img_path = os.path.join(output_folder, img_file)  # Save with the original filename
        cv2.imwrite(output_img_path, image)
        print(f"✅ Processed: {img_file} → Saved to {output_folder}")

# Save evaluation results to CSV
df = pd.DataFrame(evaluation_data)
df.to_csv(csv_path, index=False)
print(f"✅ Evaluation results saved in: {csv_path}")
