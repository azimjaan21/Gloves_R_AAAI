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
csv_path = os.path.join(output_folder, "models/hybrid_results.csv")

# Define wrist keypoint indexes (from YOLO-Pose keypoint order)
WRIST_KEYPOINTS = [9, 10]  # Right wrist (9), Left wrist (10)
DISTANCE_THRESHOLD = 25  # Pixels for wrist-glove matching

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

        # Extract glove segmentation masks and classify by type
        glove_masks, no_glove_masks = [], []
        for result in seg_results:
            if result.masks is not None:
                for i, mask in enumerate(result.masks.xy):
                    class_id = int(result.boxes.cls[i].item())  # Get class ID
                    if class_id == 0:
                        glove_masks.append(mask.astype(np.int32))  # Gloves (Class 0)
                    elif class_id == 1:
                        no_glove_masks.append(mask.astype(np.int32))  # No Gloves (Class 1)

        # Perform keypoint-based filtering
        valid_glove_masks, valid_no_glove_masks = [], []

        for mask_pts in glove_masks:
            if any(
                cv2.pointPolygonTest(mask_pts, wrist, False) >= 0 or is_wrist_near_glove(wrist, mask_pts)
                for wrist in wrist_keypoints
            ):
                valid_glove_masks.append(mask_pts)

        for mask_pts in no_glove_masks:
            if any(
                cv2.pointPolygonTest(mask_pts, wrist, False) >= 0 or is_wrist_near_glove(wrist, mask_pts)
                for wrist in wrist_keypoints
            ):
                valid_no_glove_masks.append(mask_pts)

        # Count metrics
        total_gloves_detected = len(glove_masks)
        total_no_gloves_detected = len(no_glove_masks)
        valid_gloves = len(valid_glove_masks)
        valid_no_gloves = len(valid_no_glove_masks)

        false_positives_gloves = total_gloves_detected - valid_gloves
        false_positives_no_gloves = total_no_gloves_detected - valid_no_gloves

        precision_gloves = valid_gloves / (valid_gloves + false_positives_gloves + 1e-6)  # Avoid div by zero
        precision_no_gloves = valid_no_gloves / (valid_no_gloves + false_positives_no_gloves + 1e-6)

        # Store evaluation data
        evaluation_data.append({
            "Image": img_file,
            "Total Gloves Detected": total_gloves_detected,
            "Valid Gloves After Filtering": valid_gloves,
            "Total No Gloves Detected": total_no_gloves_detected,
            "Valid No Gloves After Filtering": valid_no_gloves,
            "False Positives Gloves": false_positives_gloves,
            "False Positives No Gloves": false_positives_no_gloves,
            "Precision Gloves": round(precision_gloves, 4),
            "Precision No Gloves": round(precision_no_gloves, 4)
        })

        # Visualization
        for mask_pts in glove_masks:
            cv2.polylines(image, [mask_pts], isClosed=True, color=(255, 0, 0), thickness=2)  # Blue outline for Gloves
        for mask_pts in no_glove_masks:
            cv2.polylines(image, [mask_pts], isClosed=True, color=(0, 165, 255), thickness=2)  # Orange outline for No Gloves

        for mask_pts in valid_glove_masks:
            cv2.fillPoly(image, [mask_pts], color=(0, 255, 0))  # Green for valid Gloves
        for mask_pts in valid_no_glove_masks:
            cv2.fillPoly(image, [mask_pts], color=(0, 0, 255))  # Red for valid No Gloves

        for wrist_x, wrist_y in wrist_keypoints:
            cv2.circle(image, (wrist_x, wrist_y), 6, (180, 105, 255), -1)  # Pink wrist keypoints

        # Save processed image
        output_img_path = os.path.join(output_folder, img_file)
        cv2.imwrite(output_img_path, image)

# Save evaluation results
df = pd.DataFrame(evaluation_data)
df.to_csv(csv_path, index=False)
print(f"✅ Evaluation results saved in: {csv_path}")
