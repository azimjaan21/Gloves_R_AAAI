import cv2
import numpy as np

image_path = "C:/Users/dalab/Desktop/azimjaan21/Gloves_R_AAAI/no_gloves_augmented/train/images/sample.jpg"
label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")

# Read image
img = cv2.imread(image_path)

# Read YOLO labels
with open(label_path, "r") as file:
    for line in file.readlines():
        values = list(map(float, line.strip().split()))
        cls = int(values[0])
        points = np.array(values[1:]).reshape(-1, 2) * np.array([img.shape[1], img.shape[0]])

        # Draw mask
        cv2.polylines(img, [points.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

cv2.imshow("Augmented Image with Labels", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
