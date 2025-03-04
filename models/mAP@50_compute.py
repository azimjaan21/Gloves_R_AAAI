import numpy as np
import glob
from collections import defaultdict

# Load Ground Truth
def load_ground_truth(gt_folder):
    gt_boxes = defaultdict(list)
    for file in glob.glob(gt_folder + "/*.txt"):
        img_id = file.split("/")[-1].replace(".txt", "")
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                x, y, w, h = parts[1], parts[2], parts[3], parts[4]
                gt_boxes[img_id].append([x, y, w, h])
    return gt_boxes

# Load Hybrid Model Detections
def load_detections(det_folder):
    det_boxes = defaultdict(list)
    for file in glob.glob(det_folder + "/*.txt"):
        img_id = file.split("/")[-1].replace(".txt", "")
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                conf, x, y, w, h = parts[1], parts[2], parts[3], parts[4], parts[5]
                det_boxes[img_id].append([conf, x, y, w, h])
    return det_boxes

# Compute IoU
def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Compute mAP
def compute_map(gt_boxes, det_boxes, iou_thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]):
    ap_values = []
    for iou_thresh in iou_thresholds:
        tp, fp, fn = 0, 0, 0
        for img_id in gt_boxes:
            gt = gt_boxes[img_id]
            det = det_boxes.get(img_id, [])
            matched = set()
            for d in det:
                conf, dx, dy, dw, dh = d
                best_iou, best_gt_idx = 0, -1
                for i, g in enumerate(gt):
                    if i in matched:
                        continue
                    iou_score = iou(g, [dx, dy, dw, dh])
                    if iou_score > best_iou:
                        best_iou, best_gt_idx = iou_score, i
                if best_iou >= iou_thresh:
                    tp += 1
                    matched.add(best_gt_idx)
                else:
                    fp += 1
            fn += len(gt) - len(matched)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        ap = (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        ap_values.append(ap)
    mAP50, mAP5095 = ap_values[0], np.mean(ap_values)
    return mAP50, mAP5095

# Define paths
gt_folder = "path/to/valid/labels"  # Ground truth labels
det_folder = "path/to/hybrid_model_detections"  # Hybrid model filtered detections

# Load data
gt_boxes = load_ground_truth(gt_folder)
det_boxes = load_detections(det_folder)

# Compute mAP
mAP50, mAP5095 = compute_map(gt_boxes, det_boxes)

# Print results
print(f"mAP@50: {mAP50:.4f}")
print(f"mAP@50-95: {mAP5095:.4f}")
