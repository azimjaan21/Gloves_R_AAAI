from ultralytics import YOLO
import cv2
import numpy as np

# Load trained model
model = YOLO(r'runs/segment/train/weights/gloves.pt')  # Path to trained model

# Define source (change to test on different media)
input_source = "electrical.mp4"  # Change to "image.jpg" or "video.mp4"
is_video = input_source.endswith((".mp4", ".avi", ".mov"))

# Open video file or read image
if is_video:
    cap = cv2.VideoCapture(input_source)
else:
    image = cv2.imread(input_source)

# Define mask color
mask_color = (0, 255, 0)  # Green in BGR

while is_video and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(frame, task="segment", conf=0.5)

    # Create an overlay for masks
    mask_overlay = frame.copy()

    for result in results:
        masks = result.masks
        if masks is not None:
            for i, mask in enumerate(masks.xy):
                mask = np.array(mask, dtype=np.int32)
                cv2.fillPoly(mask_overlay, [mask], mask_color)

                # Get label and confidence score
                label = result.names[result.boxes.cls[i].item()]
                conf = result.boxes.conf[i].item()

                # Display label & confidence
                x, y = mask[0][0], mask[0][1]
                text = f"{label} {conf:.2f}"
                cv2.putText(mask_overlay, text, (int(x), int(y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Blend mask overlay with frame
    final_output = cv2.addWeighted(mask_overlay, 0.5, frame, 0.5, 0)

    # Show output
    cv2.imshow("Segmentation Output", final_output)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# If processing an image
if not is_video:
    results = model.predict(image, task="segment", conf=0.25)
    mask_overlay = image.copy()

    for result in results:
        masks = result.masks
        if masks is not None:
            for i, mask in enumerate(masks.xy):
                mask = np.array(mask, dtype=np.int32)
                cv2.fillPoly(mask_overlay, [mask], mask_color)

                # Get label and confidence score
                label = result.names[result.boxes.cls[i].item()]
                conf = result.boxes.conf[i].item()

                # Display label & confidence
                x, y = mask[0][0], mask[0][1]
                text = f"{label} {conf:.2f}"
                cv2.putText(mask_overlay, text, (int(x), int(y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Blend mask overlay with image
    final_output = cv2.addWeighted(mask_overlay, 0.5, image, 0.5, 0)

    # Show final image
    cv2.imshow("Segmentation Output", final_output)
    cv2.waitKey(0)

# Cleanup
cap.release() if is_video else None
cv2.destroyAllWindows()
