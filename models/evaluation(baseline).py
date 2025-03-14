from ultralytics import YOLO

def validate_model():
    model = YOLO(r'runs/segment/train/weights/best.pt')  # Load trained model

    model.val(
        data=r'gloves.yaml',  
        imgsz=640,  
        device='cuda', 
        conf=0.25,  
        batch=16,  
        iou=0.55,
        save=True,  
        save_json=True,  # ✅ Save results in COCO format
        save_txt=True,  # ✅ Save label outputs
        verbose=True,  # ✅ Show detailed logs
    )

if __name__ == '__main__':
    validate_model()
