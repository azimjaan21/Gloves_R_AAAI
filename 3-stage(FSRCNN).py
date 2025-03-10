import cv2
import numpy as np
import torch
from ultralytics import YOLO

# **Step 1: Define ESPCN Model**
class ESPCN_model(torch.nn.Module):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
        torch.nn.init.normal_(self.conv_1.weight, mean=0, std=0.001)
        torch.nn.init.zeros_(self.conv_1.bias)

        self.tanh = torch.nn.Tanh()

        self.conv_2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        torch.nn.init.normal_(self.conv_2.weight, mean=0, std=0.001)
        torch.nn.init.zeros_(self.conv_2.bias)

        self.conv_3 = torch.nn.Conv2d(in_channels=32, out_channels=(3 * scale * scale), kernel_size=3, padding=1)
        torch.nn.init.normal_(self.conv_3.weight, mean=0, std=0.001)
        torch.nn.init.zeros_(self.conv_3.bias)

        self.pixel_shuffle = torch.nn.PixelShuffle(scale)

    def forward(self, X_in):
        X = self.tanh(self.conv_1(X_in))
        X = self.tanh(self.conv_2(X))
        X = self.conv_3(X)
        X = self.pixel_shuffle(X)
        X_out = torch.clip(X, 0.0, 1.0)
        return X_out


# **Step 2: Load ESPCN Model**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sr_model = ESPCN_model(scale=4).to(device)
sr_model.load_state_dict(torch.load("ESPCN-x4.pt", map_location=device))
sr_model.eval()
print("✅ ESPCN Model Loaded Successfully!")

# **Step 3: Load YOLO Models (Segmentation & Pose)**
seg_model = YOLO("runs/segment/train/weights/best.pt")  # Glove Segmentation
pose_model = YOLO("yolo11m-pose.pt")  # Pose Estimation (Only for Hands)

# **Step 4: Choose Input Type**
input_source = "gloves.mp4"  # Change to "image.jpg" or "video.mp4"
is_video = input_source.endswith((".mp4", ".avi", ".mov"))

cap = cv2.VideoCapture(input_source if is_video else 0)

# Wrist keypoint indexes in YOLO-Pose
WRIST_INDEXES = [9, 10]  # Left wrist = 9, Right wrist = 10
DISTANCE_THRESHOLD = 30  # Pixels for wrist-glove matching

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # **Step 5: Apply ESPCN Super-Resolution**
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame_rgb = frame_rgb.astype(np.float32) / 255.0  # Normalize to [0,1]
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # Convert to tensor

    with torch.no_grad():
        enhanced_frame = sr_model(frame_tensor)

    # Convert Back to Image
    enhanced_frame = enhanced_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_frame = (enhanced_frame * 255.0).clip(0, 255).astype(np.uint8)
    enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # **Step 6: Run YOLO Inference on Enhanced Image**
    glove_detections = seg_model.predict(enhanced_frame, task="segment", conf=0.25)
    pose_detections = pose_model.predict(enhanced_frame, task="pose", conf=0.25)

    mask_overlay = enhanced_frame.copy()  # Create an overlay for masks

    # **Step 7: Extract Wrist Keypoints**
    wrist_keypoints = []
    for result in pose_detections:
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            for person in result.keypoints.data:
                if len(person) > max(WRIST_INDEXES):
                    left_wrist = person[9][:2] if not torch.isnan(person[9][:2]).any() else None
                    right_wrist = person[10][:2] if not torch.isnan(person[10][:2]).any() else None

                    if left_wrist is not None:
                        left_wrist = tuple(map(int, left_wrist.cpu().numpy()))
                    if right_wrist is not None:
                        right_wrist = tuple(map(int, right_wrist.cpu().numpy()))

                    wrist_keypoints.append((left_wrist, right_wrist))

    # **Step 8: Process Glove Detections**
    for result in glove_detections:
        if result.masks is not None and len(result.masks.xy) > 0:
            for i, mask in enumerate(result.masks.xy):
                mask_poly = np.array(mask, dtype=np.int32)
                glove_conf = result.boxes.conf[i].cpu().numpy() if result.boxes is not None else 0.0

                # Find the closest side of the glove to the wrist
                is_matched = any(
                    min(np.linalg.norm(np.array(wrist) - np.array(mask_pt)) for mask_pt in mask_poly) < DISTANCE_THRESHOLD
                    for wrist_pair in wrist_keypoints
                    for wrist in wrist_pair if wrist is not None
                )

                # Only show matched gloves (Filtered gloves are NOT displayed)
                if is_matched:
                    cv2.fillPoly(mask_overlay, [mask_poly], color=(0, 255, 0))  # Green overlay fill
                    cv2.polylines(mask_overlay, [mask_poly], isClosed=True, color=(0, 255, 0), thickness=2)

                    # Show class label & confidence
                    text = f"Gloves {glove_conf:.2f}"
                    cv2.putText(mask_overlay, text, (mask_poly[0][0], mask_poly[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # **Step 9: Blend Overlay & Draw Keypoints**
    frame = cv2.addWeighted(mask_overlay, 0.5, enhanced_frame, 0.5, 0)

    # Draw wrist keypoints (Now in **RED**)
    for (lwrist, rwrist) in wrist_keypoints:
        if lwrist is not None:
            cv2.circle(frame, lwrist, 6, (0, 0, 255), -1)  # Red for left wrist
        if rwrist is not None:
            cv2.circle(frame, rwrist, 6, (0, 0, 255), -1)  # Red for right wrist

    # **Step 10: Show Output**
    cv2.imshow("Glove Detection (Super-Resolution Enhanced)", frame)

    # Press 'q' to exit video processing
    if is_video:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKey(0)
        break

cap.release()
cv2.destroyAllWindows()
