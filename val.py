from ultralytics import YOLO

def validate_model():
    model = YOLO(r'runs/segment/train/weights/best.pt')  # Load trained model

    model.val(
        data=r'gloves.yaml',  
        imgsz=640,  
        device='cuda', 
        conf=0.25,  
        iou=0.65,  
        batch=16,  
        save=True,  
        save_json=True,  # ✅ Save results in COCO format
        save_txt=True,  # ✅ Save label outputs
        verbose=True,  # ✅ Show detailed logs
        half=True,  # ✅ Use FP16 precision for faster inference
        split='val'  # ✅ Ensure we evaluate on the validation dataset
    )

if __name__ == '__main__':
    validate_model()
