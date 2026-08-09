
import pandas as pd
import cv2
import os
import random
import yaml

KAGGLE_ROOT = "data/external/bdd100k_kaggle"
VIDEO_DIR = os.path.join(KAGGLE_ROOT, "bdd100k_videos_train_00/bdd100k/videos/train")
CSV_PATH = os.path.join(KAGGLE_ROOT, "mot_labels.csv")

OUTPUT_ROOT = "data/yolo_bdd"
FRAMES_PER_VIDEO = 8       # nb d'images extraites par vidéo
CLASS_MAP = {"car": 0, "motorcycle": 1}
VAL_RATIO = 0.15
random.seed(42)

def yolo_line(cls_id, x1, y1, x2, y2, img_w, img_h):
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

print("Chargement du CSV...")
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df[df['category'].isin(CLASS_MAP.keys())]

available_videos = set(f.replace(".mov", "") for f in os.listdir(VIDEO_DIR) if f.endswith(".mov"))
print(f"Vidéos disponibles localement : {len(available_videos)}")

df = df[df['videoName'].isin(available_videos)]
video_list = sorted(df['videoName'].unique())
random.shuffle(video_list)
print(f"Vidéos utilisables (avec car/moto) : {len(video_list)}")

n_val = int(len(video_list) * VAL_RATIO)
val_videos = set(video_list[:n_val])
train_videos = set(video_list[n_val:])

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

total_images = 0
for video_name in video_list:
    split = "val" if video_name in val_videos else "train"
    video_path = os.path.join(VIDEO_DIR, f"{video_name}.mov")
    sub = df[df['videoName'] == video_name]

    frame_indices = sorted(sub['frameIndex'].unique())
    if len(frame_indices) < FRAMES_PER_VIDEO:
        chosen = frame_indices
    else:
        step = len(frame_indices) // FRAMES_PER_VIDEO
        chosen = frame_indices[::step][:FRAMES_PER_VIDEO]

    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for label_idx in chosen:
        target_frame = round((label_idx / 5.0) * real_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        img_name = f"{video_name}_{int(label_idx):04d}.jpg"
        img_path = os.path.join(OUTPUT_ROOT, "images", split, img_name)
        cv2.imwrite(img_path, frame)

        rows = sub[sub['frameIndex'] == label_idx]
        label_path = os.path.join(OUTPUT_ROOT, "labels", split, img_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for _, row in rows.iterrows():
                cls_id = CLASS_MAP[row['category']]
                line = yolo_line(cls_id, row['box2d.x1'], row['box2d.y1'],
                                  row['box2d.x2'], row['box2d.y2'], img_w, img_h)
                f.write(line + "\n")
        total_images += 1

    cap.release()
    if total_images % 200 == 0:
        print(f"... {total_images} images traitées")

data_yaml = {
    "path": os.path.abspath(OUTPUT_ROOT),
    "train": "images/train",
    "val": "images/val",
    "names": {0: "car", 1: "motorcycle"}
}
with open(os.path.join(OUTPUT_ROOT, "data.yaml"), "w") as f:
    yaml.dump(data_yaml, f)

print(f"\nTerminé. Total images : {total_images}")
print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")