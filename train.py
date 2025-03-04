import torch
from ultralytics import YOLO

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # ✅ Fix for Windows multiprocessing

    # Load YOLOv11-Seg Model
    model = YOLO("yolo11m-seg.pt")

    # Train Model
    model.train(
        data="gloves.yaml",  # Dataset configuration file
        epochs=100,  # Total training epochs
        imgsz=640,  # Image size
        batch=16,  # Batch size (adjust based on GPU)
        lr0=0.002,  # Learning rate
        optimizer="AdamW",  # Optimizer (AdamW is stable for segmentation)
        cos_lr=True,  # Cosine learning rate decay
        augment=True,  # Enable augmentations
        mosaic=0.7,  # Mosaic augmentation
        mixup=0.2,  # Mixup augmentation
        copy_paste=0.4,  # Copy-Paste augmentation for better segmentation
        flipud=0.2,  # Vertical flip
        fliplr=0.5,  # Horizontal flip
        scale=0.5,  # Scale augmentation
        translate=0.2,  # Translation augmentation
        shear=0.1,  # Shear augmentation
        iou=0.55,  # IoU threshold (higher = better precision)
        nms=True,  # Non-Maximum Suppression threshold
        label_smoothing=0.1,  # Helps balance class learning
        val=True,  # Run validation
        device='cuda'  # Use GPU (or set to 'cpu' if no GPU available)
    )

    print("✅ Training Completed!")
