from ultralytics import YOLO

def validate_model():
    model = YOLO(r'runs/segment/train/weights/best.pt')  

    model.val(
        data=r'gloves.yaml',  
        imgsz=640,
        device='cuda',  
        conf=0.25,
        task="segment",  # Segmentation mode
        verbose=True,
        save=True
    )

if __name__ == '__main__':
    validate_model()
