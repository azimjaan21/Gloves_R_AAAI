import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load Hybrid Model Components
pose_model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\yolo11m-pose.pt")  # Wrist detection
segmentation_model = YOLO(r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\runs\segment\train\weights\gloves.pt")  # Glove segmentation

# Dataset Paths
dataset_path = r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\old_evaluation2"
image_folder = os.path.join(dataset_path, "images")
label_folder = os.path.join(dataset_path, "labels")  # Ground Truth Labels

output_folder = r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\hybrid_results"
os.makedirs(output_folder, exist_ok=True)

# CSV file for results
csv_path = os.path.join(output_folder, "hybrid_evaluation_results.csv")

# Wrist keypoint indexes in YOLOv11-Pose
WRIST_INDEXES = [9, 10]  # Left wrist = 9, Right wrist = 10
DISTANCE_THRESHOLD = 50  # Pixels for wrist-glove matching
IOU_THRESHOLD = 0.5 # IoU Threshold for Matching

# Function to compute shortest distance from wrist to closest glove polygon side
def point_to_line_distance(point, line_start, line_end):
    """Computes the shortest Euclidean distance between a point and a line segment."""
    line = np.array(line_end) - np.array(line_start)
    if np.dot(line, line) == 0:
        return np.linalg.norm(point - line_start)  # If same points, return direct distance

    t = max(0, min(1, np.dot(point - line_start, line) / np.dot(line, line)))
    projection = line_start + t * line  # Projection on the segment
    return np.linalg.norm(point - projection)

# Function: Check if wrist keypoint is inside or near a mask polygon
def is_wrist_near_glove(wrist, mask_pts, threshold=DISTANCE_THRESHOLD):
    """Check if wrist keypoint is inside or near a glove/no-glove mask polygon."""
    if cv2.pointPolygonTest(mask_pts, wrist, False) >= 0:
        return True  # Wrist is inside the mask polygon
    for i in range(len(mask_pts) - 1):
        if point_to_line_distance(wrist, mask_pts[i], mask_pts[i + 1]) < threshold:
            return True  # Wrist is near the polygon edge
    return False

# Function: Compute IoU for Mask Matching
def compute_iou(pred_mask, gt_mask, image_shape=(640, 640)):
    """Computes IoU (Intersection over Union) between two polygon masks."""
    pred_binary_mask = np.zeros(image_shape, dtype=np.uint8)
    gt_binary_mask = np.zeros(image_shape, dtype=np.uint8)

    cv2.fillPoly(pred_binary_mask, [np.array(pred_mask, dtype=np.int32)], 1)
    cv2.fillPoly(gt_binary_mask, [np.array(gt_mask, dtype=np.int32)], 1)

    intersection = np.logical_and(pred_binary_mask, gt_binary_mask).sum()
    union = np.logical_or(pred_binary_mask, gt_binary_mask).sum()

    return intersection / (union + 1e-6)  # Avoid division by zero

# Store evaluation results
evaluation_data = []

# Process Each Image in Dataset
for img_file in os.listdir(image_folder):
    if not img_file.endswith(('.jpg', '.png', '.jpeg')):
        continue  # Skip non-image files

    img_path = os.path.join(image_folder, img_file)
    image = cv2.imread(img_path)
    overlay = image.copy()

    # Run YOLO Pose & Segmentation
    pose_results = pose_model.predict(img_path, task="pose", device="cuda", conf=0.25, save=False)
    seg_results = segmentation_model.predict(img_path, task="segment", device="cuda", conf=0.25, save=False)

    # Extract Wrist Keypoints
    wrist_keypoints = []
    for result in pose_results:
        if result.keypoints is not None:
            for kp in result.keypoints.xy:
                if len(kp) > max(WRIST_INDEXES):  # Ensure keypoints exist
                    wrist_keypoints.append((int(kp[9][0]), int(kp[9][1])))  # Right wrist
                    wrist_keypoints.append((int(kp[10][0]), int(kp[10][1])))  # Left wrist

    # Extract Predicted Glove & No-Glove Masks
    glove_masks = []
    no_glove_masks = []
    for result in seg_results:
        if result.masks is not None:
            for mask, cls in zip(result.masks.xy, result.boxes.cls):
                mask_pts = mask.astype(np.int32)
                if int(cls) == 0:
                    glove_masks.append(mask_pts)  # Class 0: Gloves
                else:
                    no_glove_masks.append(mask_pts)  # Class 1: No Gloves

    # Validate Gloves & No Gloves using Wrist Keypoints
    valid_glove_masks = [mask for mask in glove_masks if any(is_wrist_near_glove(wrist, mask) for wrist in wrist_keypoints)]
    valid_no_glove_masks = [mask for mask in no_glove_masks if any(is_wrist_near_glove(wrist, mask) for wrist in wrist_keypoints)]



    # Draw Gloves (Green) and No Gloves (Red) with 50% Opacity
    for mask in glove_masks:
        cv2.fillPoly(overlay, [mask], (0, 255, 0))  # Green for gloves
    for mask in no_glove_masks:
        cv2.fillPoly(overlay, [mask], (0, 0, 255))  # Red for no gloves

            # Draw wrist keypoints (Pink)
    for wrist_x, wrist_y in wrist_keypoints:
        cv2.circle(overlay, (wrist_x, wrist_y), 6, (255, 0, 0), -1)

    # Blend overlay with transparency
    image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    # Save Processed Image
    output_img_path = os.path.join(output_folder, img_file)
    cv2.imwrite(output_img_path, image)

    # Compute Metrics
    total_gloves_detected = len(glove_masks)
    valid_gloves = len(valid_glove_masks)
    false_positives_gloves = total_gloves_detected - valid_gloves

    total_no_gloves_detected = len(no_glove_masks)
    valid_no_gloves = len(valid_no_glove_masks)
    false_positives_no_gloves = total_no_gloves_detected - valid_no_gloves

    precision_gloves = valid_gloves / (valid_gloves + false_positives_gloves + 1e-6)
    precision_no_gloves = valid_no_gloves / (valid_no_gloves + false_positives_no_gloves + 1e-6)

    # Store Evaluation Data
    evaluation_data.append({
        "Image": img_file,
        "Total Gloves Detected": total_gloves_detected,
        "Valid Gloves": valid_gloves,
        "Total No Gloves Detected": total_no_gloves_detected,
        "Valid No Gloves": valid_no_gloves,
        "Precision Gloves": round(precision_gloves, 4),
        "Precision No Gloves": round(precision_no_gloves, 4)
    })

print(f"✅ Processed all images. Results saved to {output_folder}")

# Save Evaluation Results to CSV
df = pd.DataFrame(evaluation_data)
df.to_csv(csv_path, index=False)
print(f"✅ Evaluation results saved in: {csv_path}")
