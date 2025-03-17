import os

# Dataset Path
label_folder = r"C:\Users\dalab\Desktop\azimjaan21\Gloves_R_AAAI\old_evaluation2\labels"

# Initialize Counters
gloves_count = 0
no_gloves_count = 0

# Read all label files in the dataset
for label_file in os.listdir(label_folder):
    if label_file.endswith(".txt"):
        with open(os.path.join(label_folder, label_file), "r") as file:
            for line in file:
                class_id = int(line.strip().split()[0])  # Get first value (Class ID)
                if class_id == 0:
                    gloves_count += 1
                elif class_id == 1:
                    no_gloves_count += 1

# Print Results
print(f"🟢 Total Ground Truth Gloves: {gloves_count}")
print(f"🔴 Total Ground Truth No Gloves: {no_gloves_count}")
