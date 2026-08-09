
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # version "nano", la plus légère, adaptée au CPU

model.train(
    data="data/yolo_bdd/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    device="cpu",
    workers=2,
    project="runs/detect",
    name="bdd_car_moto"
)