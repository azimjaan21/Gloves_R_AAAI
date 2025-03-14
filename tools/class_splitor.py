import os
import shutil
from tqdm import tqdm

# Define dataset base path
base_path = "C:/Users/dalab/Desktop/azimjaan21/Gloves_R_AAAI/dataset_333"
output_path = "C:/Users/dalab/Desktop/azimjaan21/Gloves_R_AAAI/no_gloves_extracted"

# Define class ID for "No Gloves" in YOLO format (Update if different)
NO_GLOVES_CLASS_ID = "1"  # Assuming "1" represents "No Gloves" class in annotation files

# Dataset subsets
subsets = ["train", "valid"]

for subset in subsets:
    label_path = os.path.join(base_path, subset, "labels")
    image_path = os.path.join(base_path, subset, "images")
    
    output_label_path = os.path.join(output_path, subset, "labels")
    output_image_path = os.path.join(output_path, subset, "images")
    
    os.makedirs(output_label_path, exist_ok=True)
    os.makedirs(output_image_path, exist_ok=True)
    
    # Process each label file
    for label_file in tqdm(os.listdir(label_path), desc=f"Processing {subset}"):
        if label_file.endswith(".txt"):
            label_filepath = os.path.join(label_path, label_file)
            
            # Read the label file to check if it contains "No Gloves" class
            with open(label_filepath, "r") as f:
                labels = f.readlines()
            
            contains_no_gloves = any(NO_GLOVES_CLASS_ID in line.split()[0] for line in labels)

            if contains_no_gloves:
                # Copy label file
                shutil.copy(label_filepath, os.path.join(output_label_path, label_file))

                # Copy corresponding image
                image_filename_jpg = label_file.replace(".txt", ".jpg")
                image_filename_png = label_file.replace(".txt", ".png")

                # Check if the image exists and copy it
                if os.path.exists(os.path.join(image_path, image_filename_jpg)):
                    shutil.copy(os.path.join(image_path, image_filename_jpg), os.path.join(output_image_path, image_filename_jpg))
                elif os.path.exists(os.path.join(image_path, image_filename_png)):
                    shutil.copy(os.path.join(image_path, image_filename_png), os.path.join(output_image_path, image_filename_png))

# Count extracted files
extracted_train_images = len(os.listdir(os.path.join(output_path, "train", "images")))
extracted_train_labels = len(os.listdir(os.path.join(output_path, "train", "labels")))
extracted_valid_images = len(os.listdir(os.path.join(output_path, "valid", "images")))
extracted_valid_labels = len(os.listdir(os.path.join(output_path, "valid", "labels")))

print(f"✅ Extraction Complete!")
print(f"📂 Train: {extracted_train_images} images & {extracted_train_labels} labels extracted.")
print(f"📂 Valid: {extracted_valid_images} images & {extracted_valid_labels} labels extracted.")
