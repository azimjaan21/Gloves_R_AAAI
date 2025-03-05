import cv2
import numpy as np
import torch
from ultralytics import YOLO
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Load YOLO models
seg_model = YOLO("runs/segment/train/weights/best.pt")  # Glove Segmentation
pose_model = YOLO("yolo11m-pose.pt")  # Pose Estimation (Only for Hands)

# Load Real-ESRGAN model
model_path = "weights/RealESRGAN_x4plus.pth"
state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))['params_ema']
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.load_state_dict(state_dict, strict=True)
upsampler = RealESRGANer(
    scale=4, model_path=model_path, model=model, tile=0, pre_pad=0, half=True if torch.cuda.is_available() else False
)

# Choose input source
input_source = "tekpe.mp4"
is_video = input_source.endswith(('.mp4', '.avi', '.mov'))
cap = cv2.VideoCapture(input_source if is_video else 0)

# Wrist keypoint indexes in YOLOv11-Pose
WRIST_INDEXES = [9, 10]  # Left wrist = 9, Right wrist = 10
DISTANCE_THRESHOLD = 25  # Pixels for wrist-glove matching

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply Real-ESRGAN super-resolution to enhance the frame
    frame, _ = upsampler.enhance(frame, outscale=4)
    
    # Create an overlay for semi-transparent masks
    mask_overlay = frame.copy()
    
    # Run inference
    glove_detections = seg_model.predict(frame, task="segment", conf=0.25)
    pose_detections = pose_model.predict(frame, task="pose", conf=0.25)
    
    # Extract wrist keypoints
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

    # Process glove detections (Apply wrist-side filtering)
    for result in glove_detections:
        if result.masks is not None and len(result.masks.xy) > 0:
            for i, mask in enumerate(result.masks.xy):
                mask_poly = np.array(mask, dtype=np.int32)
                glove_conf = result.boxes.conf[i].cpu().numpy() if result.boxes is not None else 0.0
                is_matched = any(
                    min(np.linalg.norm(np.array(wrist) - np.array(mask_pt)) for mask_pt in mask_poly) < DISTANCE_THRESHOLD
                    for wrist_pair in wrist_keypoints
                    for wrist in wrist_pair if wrist is not None
                )
                if is_matched:
                    cv2.fillPoly(mask_overlay, [mask_poly], color=(0, 255, 0))
                    cv2.polylines(frame, [mask_poly], isClosed=True, color=(0, 255, 0), thickness=2)
                    text = f"Gloves {glove_conf:.2f}"
                    cv2.putText(frame, text, (mask_poly[0][0], mask_poly[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Blend overlay with 50% transparency
    frame = cv2.addWeighted(mask_overlay, 0.5, frame, 0.5, 0)
    
    # Draw wrist keypoints
    for (lwrist, rwrist) in wrist_keypoints:
        if lwrist is not None:
            cv2.circle(frame, lwrist, 6, (0, 0, 255), -1)
        if rwrist is not None:
            cv2.circle(frame, rwrist, 6, (0, 0, 255), -1)
    
    # Show output
    cv2.imshow("Glove Detection", frame)
    
    if is_video:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.waitKey(0)
        break

cap.release()
cv2.destroyAllWindows()