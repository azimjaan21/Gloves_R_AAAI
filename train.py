from ultralytics import YOLO

def train_model():
    model = YOLO('yolo11m-seg.pt')  

    model.train(
        data=r'gloves.yaml',  
        epochs=50,
        batch=8,
        imgsz=640,
        device='cuda',  
        lr0=0.01,  # Learning rate
        weight_decay=0.0005,  # Regularization
        patience=10,  # Early stopping
        optimizer='SGD',
        momentum=0.937,
        augment=True,  
        plots=True,  
        verbose=True,
        task="segment" 
    )

if __name__ == '__main__':
    train_model()
