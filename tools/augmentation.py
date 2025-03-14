import albumentations as A
import cv2
import os
import numpy as np
from tqdm import tqdm

# Base paths
input_base = "C:/Users/dalab/Desktop/azimjaan21/Gloves_R_AAAI/no_gloves_extracted"
output_base = "C:/Users/dalab/Desktop/azimjaan21/Gloves_R_AAAI/no_gloves_augmented"

# Ensure output directories exist
for subset in ["train", "valid"]:
    os.makedirs(os.path.join(output_base, subset, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, subset, "labels"), exist_ok=True)

# Augmentation pipeline (Images + YOLO Labels)
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Perspective(scale=(0.015, 0.03), p=0.3),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=8, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))  # Ensures YOLO labels are updated

# Number of augmented images per original image
augment_per_image = 3  

# Process each subset (train & valid)
for subset in ["train", "valid"]:
    image_folder = os.path.join(input_base, subset, "images")
    label_folder = os.path.join(input_base, subset, "labels")
    
    output_image_folder = os.path.join(output_base, subset, "images")
    output_label_folder = os.path.join(output_base, subset, "labels")
    
    # Process each image
    for filename in tqdm(os.listdir(image_folder), desc=f"Augmenting {subset}"):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

            img = cv2.imread(img_path)
            if img is None or not os.path.exists(label_path):
                continue  # Skip corrupted files or missing labels

            # Read YOLO segmentation labels (polygon format)
            bboxes = []
            class_labels = []
            with open(label_path, "r") as file:
                for line in file.readlines():
                    values = list(map(float, line.strip().split()))
                    class_labels.append(int(values[0]))  # Class ID
                    bboxes.append(values[1:])  # Polygon points

            # Apply augmentations multiple times per image
            for i in range(augment_per_image):
                augmented = augmentation_pipeline(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_img, aug_bboxes = augmented["image"], augmented["bboxes"]

                output_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
                output_labelname = f"{os.path.splitext(filename)[0]}_aug_{i}.txt"

                # Save augmented image
                cv2.imwrite(os.path.join(output_image_folder, output_filename), aug_img)

                # Save updated YOLO segmentation labels
                with open(os.path.join(output_label_folder, output_labelname), "w") as label_file:
                    for cls, bbox in zip(class_labels, aug_bboxes):
                        label_file.write(f"{cls} " + " ".join(map(str, bbox)) + "\n")

# Count augmented images
augmented_train_images = len(os.listdir(os.path.join(output_base, "train", "images")))
augmented_valid_images = len(os.listdir(os.path.join(output_base, "valid", "images")))

print(f"✅ Augmentation Complete!")
print(f"📂 Train: {augmented_train_images} augmented images.")
print(f"📂 Valid: {augmented_valid_images} augmented images.")
