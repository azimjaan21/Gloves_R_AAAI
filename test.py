import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load trained model
model = YOLO(r'runs/segment/train/weights/best.pt')  # Path to trained model

# Create output directory
output_dir = "results_segmentation"
os.makedirs(output_dir, exist_ok=True)

# Test image path
image_path = "test_data/test6.jpg"
output_path = os.path.join(output_dir, os.path.basename(image_path))  # Save in output_results/

# Run inference
results = model.predict(
    source=image_path,  # Path to test image
    task="segment",  # Enable segmentation mode
    conf=0.65,  # Confidence threshold
    save=True,  # Save outputs
    show=False  # Don't use built-in display
)

# Read the original image
image = cv2.imread(image_path)
overlay = image.copy()  # Copy for transparency effect

# Define colors for masks and outlines
colors = {
    0: {"mask": (0, 255, 0), "outline": (0, 200, 0)},  # Green for gloves
    1: {"mask": (0, 0, 255), "outline": (0, 0, 200)}  # Red for no_gloves
}

for result in results:
    masks = result.masks  # Get segmentation masks
    if masks is not None:
        for i, mask in enumerate(masks.xy):  # Iterate over detected masks
            mask = np.array(mask, dtype=np.int32)
            class_id = int(result.boxes.cls[i].item())  # Get class ID

            # Get correct mask and outline colors
            mask_color = colors.get(class_id, {"mask": (255, 255, 255), "outline": (255, 255, 255)})["mask"]
            outline_color = colors.get(class_id, {"mask": (255, 255, 255), "outline": (255, 255, 255)})["outline"]

            # Apply 50% transparency to the mask
            cv2.fillPoly(overlay, [mask], mask_color)
            cv2.polylines(image, [mask], isClosed=True, color=outline_color, thickness=2)

            # Get label and confidence score
            label = result.names[class_id]
            conf = result.boxes.conf[i].item()

            # Get text position (top-left of mask)
            x, y = mask[0][0], mask[0][1]
            text = f"{label} {conf:.2f}"
            cv2.putText(image, text, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Blend the overlay with original image (50% opacity)
cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

# Save the final segmented image
cv2.imwrite(output_path, image)

print(f"✅ Segmented image saved at: {output_path}")

# Show the final image with masks and labels
cv2.imshow("Segmented Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
