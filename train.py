import torch
from ultralytics import YOLO

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  

    # Load YOLOv11-Seg Model
    model = YOLO("yolo11m-seg.pt")

    # Train Model
    model.train(
        data="gloves.yaml", 
        epochs=150,  
        imgsz=640,  
        batch=16,  
        lr0=0.0025,  # Learning rate
        optimizer="AdamW",  # Optimizer (AdamW is stable for segmentation)
        weight_decay=0.0005,  
        cos_lr=True,  
        augment=False,  
        iou=0.65,  
        nms=True,  
        label_smoothing=0.1, 
        freeze=10,  # ✅ Freezes first 10 layers for better fine-tuning
        val=True,  
        device='cuda'  # Use GPU (or set to 'cpu' if no GPU available)
    )

    print("✅ Training Completed!")
