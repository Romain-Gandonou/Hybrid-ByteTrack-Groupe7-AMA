
from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/detect/bdd_car_moto/weights/best.pt"
SOURCE = "/home/romain/Bureau/IA_Projects/Dataset_Benin_MOT/videos_originales/Video7_extrait.mp4"

model = YOLO(MODEL_PATH)

for tag, tracker_cfg in [("baseline", "bytetrack.yaml"), ("hybride", "botsort.yaml")]:
    model.track(
        source=SOURCE,
        tracker=tracker_cfg,
        persist=True,
        conf=0.25,
        iou=0.5,
        device="cpu",
        save=True,
        project="results/demo",
        name=tag,
        exist_ok=True
    )
    print(f"{tag} terminé -> results/demo/{tag}/")