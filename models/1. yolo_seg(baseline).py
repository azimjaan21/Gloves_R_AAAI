from ultralytics import YOLO
import cv2
import numpy as np

# Load trained model
model = YOLO(r'runs/segment/train/weights/best.pt')  # Path to trained model

# Run inference
results = model.predict(
    source="ww.jpg",  # Path to test image
    task="segment",  # Enable segmentation mode
    conf=0.25,  # Confidence threshold
    save=False,  # Don't save default outputs
    show=False  # Don't use built-in display
)

# Read the original image
image = cv2.imread("ww.jpg")

# Define green color for masks
mask_color = (0, 255, 0)  # Green in BGR format

for result in results:
    masks = result.masks  # Get segmentation masks
    if masks is not None:
        for i, mask in enumerate(masks.xy):  # Iterate over detected masks
            mask = np.array(mask, dtype=np.int32)
            cv2.fillPoly(image, [mask], mask_color)  # Apply green mask

            # Get label and confidence score
            label = result.names[result.boxes.cls[i].item()]
            conf = result.boxes.conf[i].item()

            # Get text position (top-left of mask)
            x, y = mask[0][0], mask[0][1]
            text = f"{label} {conf:.2f}"
            cv2.putText(image, text, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Show the final image with only green masks and labels
cv2.imshow("Segmented Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
