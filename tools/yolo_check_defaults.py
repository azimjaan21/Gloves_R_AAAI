from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")
print(model.train.__defaults__)
