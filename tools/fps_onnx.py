import time
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image

# Load ONNX model (Using GPU if available, fallback to CPU)
onnx_model_path = "realesrgan_x2.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Load test image
image_path = "cons.jpg"  # Change this to your test image
image = Image.open(image_path).convert("RGB")

# ✅ Resize image to match ONNX model input size (64x64)
image = image.resize((64, 64), Image.LANCZOS)

def enhance_image_onnx(image):
    # Preprocess image (convert to float32 and normalize)
    img_np = np.array(image).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # Change to (C, H, W) format
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension

    # Run ONNX inference with FPS measurement
    start_time = time.time()
    output = session.run(None, {"input": img_np})[0]
    end_time = time.time()

    # Calculate FPS
    fps = 1 / (end_time - start_time)
    print(f"🔥 Real-ESRGAN ONNX Inference FPS: {fps:.2f}")

    # Post-process output
    output = np.squeeze(output)  # Remove batch dimension
    output = np.transpose(output, (1, 2, 0))  # Back to (H, W, C)
    output = (output * 255.0).clip(0, 255).astype(np.uint8)  # Convert to uint8

    return Image.fromarray(output)

# Run Super-Resolution with FPS measurement
sr_image = enhance_image_onnx(image)

# Save output image
sr_image.save("output_onnx.png")
print("✅ ONNX Super-Resolution Complete!")
