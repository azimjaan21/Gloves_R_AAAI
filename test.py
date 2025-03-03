from ultralytics import YOLO

def test_model():
    model = YOLO(r'runs/train/exp/weights/best.pt')  # Path to trained model weights

    results = model.predict(
        source=r'C:\Users\dalab\Desktop\azimjaan21\SafeFactory System\glove_segmentation\test_images',  # Path to test images
        imgsz=640,
        device='cuda',  
        save=True,  # Save results
        task="segment",  # Segmentation mode
        verbose=True
    )

if __name__ == '__main__':
    test_model()
