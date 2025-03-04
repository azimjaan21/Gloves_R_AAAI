from ultralytics import YOLO
import cv2

# Load trained YOLOv11-Seg model
model = YOLO(r'runs/segment/train/weights/best.pt')  # Update path to your best model

# Run inference on an image
results = model.predict(
    source='1.jpg',  # Path to test image
    task="segment",  # Enable segmentation mode
    conf=0.5,  # Confidence threshold
    save=False,  # Don't save default YOLO output
    show=False,  # Don't use YOLO's default visualization
)

# Load image
image = cv2.imread(0)

# Define custom mask colors (Modify RGB values for different colors)
MASK_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]  # Green, Blue, Red

# Draw segmentation masks (without bounding boxes)
for idx, result in enumerate(results):
    if result.masks is not None:
        for mask in result.masks.xy:
            pts = mask.astype(int)  # Convert to integer coordinates
            color = MASK_COLORS[idx % len(MASK_COLORS)]  # Cycle through colors
            overlay = image.copy()

            # Fill mask with transparent color
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)  # Adjust transparency

# Show and save final output
cv2.imshow("Segmented Image", image)
cv2.imwrite("new1.png", image)  # Save result
cv2.waitKey(0)
cv2.destroyAllWindows()

print("✅ Inference complete! Saved as 'segmented_output.png'")