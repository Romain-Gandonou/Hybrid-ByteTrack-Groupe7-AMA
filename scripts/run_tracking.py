
from ultralytics import YOLO
import cv2, os

MODEL_PATH = "runs/detect/runs/detect/bdd_car_moto/weights/best.pt"
SEQUENCES = ["0090c713-9d58a186", "012e9465-1031243b", "00ac3256-0f8e2cda"]
TRACKERS = {"baseline": "bytetrack.yaml", "hybride": "botsort.yaml"}

model = YOLO(MODEL_PATH)

for seq in SEQUENCES:
    img_dir = f"data/BDD100K_MOT/{seq}/img1"
    frames = sorted(os.listdir(img_dir))
    h, w = cv2.imread(os.path.join(img_dir, frames[0])).shape[:2]

    tmp_video = f"/tmp/{seq}.mp4"
    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
    for f in frames:
        writer.write(cv2.imread(os.path.join(img_dir, f)))
    writer.release()

    for tag, tracker_cfg in TRACKERS.items():
        out_dir = f"results/tracking/{tag}/{seq}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "track.txt")
        lines = []
        frame_id = 0
        for r in model.track(source=tmp_video, tracker=tracker_cfg, persist=True,
                              conf=0.25, iou=0.5, device="cpu", stream=True, verbose=False):
            frame_id += 1
            if r.boxes is not None and r.boxes.id is not None:
                for box, tid in zip(r.boxes.xywh, r.boxes.id):
                    x, y, bw, bh = box.tolist()
                    x1 = x - bw/2
                    y1 = y - bh/2
                    lines.append(f"{frame_id},{int(tid)},{x1:.2f},{y1:.2f},{bw:.2f},{bh:.2f},1,-1,-1,-1")
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
        print(f"{tag} / {seq} -> {len(lines)} lignes -> {out_path}")