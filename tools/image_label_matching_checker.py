import os

dataset_path = "C:/Users/dalab/Desktop/azimjaan21/Gloves_R_AAAI/dataset_333/valid/"
image_dir = os.path.join(dataset_path, "images")
label_dir = os.path.join(dataset_path, "labels")

# Get all images and labels
image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir)}
label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir)}

# Find missing labels
missing_labels = image_files - label_files
missing_images = label_files - image_files

print(f"✅ Dataset Verification Complete!")
print(f"❌ Missing Labels: {len(missing_labels)} files")
print(f"❌ Missing Images: {len(missing_images)} files")
