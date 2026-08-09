import pandas as pd
import cv2
import os

KAGGLE_ROOT = "data/external/bdd100k_kaggle"
VIDEO_DIR = os.path.join(KAGGLE_ROOT, "bdd100k_videos_train_00/bdd100k/videos/train")
CSV_PATH = os.path.join(KAGGLE_ROOT, "mot_labels.csv")

SELECTED_VIDEOS = ["0090c713-9d58a186", "012e9465-1031243b", "00ac3256-0f8e2cda"]
CLASS_MAP = {"car": 1, "motorcycle": 2}
LABEL_FPS = 5.0  # fréquence d'échantillonnage des labels BDD100K MOT

OUTPUT_ROOT = "data/BDD100K_MOT"

def extract_sampled_frames(video_path, output_dir, n_labels):
    """Extrait uniquement les frames vidéo correspondant aux frameIndex du CSV (0 à n_labels-1)."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS)

    # frame vidéo brute cible pour chaque label i
    targets = [round((i / LABEL_FPS) * real_fps) for i in range(n_labels)]
    targets_set = set(targets)

    saved = 0
    current = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current in targets_set:
            # position de ce target dans la liste -> détermine le nouveau nom (1-indexed)
            label_positions = [i for i, t in enumerate(targets) if t == current]
            for label_i in label_positions:
                new_name = f"{label_i + 1:06d}.jpg"
                cv2.imwrite(os.path.join(output_dir, new_name), frame)
                saved += 1
        current += 1
    cap.release()
    return saved

def build_gt(df, video_name, gt_path):
    sub = df[(df['videoName'] == video_name) & (df['category'].isin(CLASS_MAP.keys()))].copy()
    with open(gt_path, 'w') as f:
        for _, row in sub.iterrows():
            frame = int(row['frameIndex']) + 1  # +1 car frameIndex démarre à 0, MOT format démarre à 1
            obj_id = int(row['id'])
            x1, y1, x2, y2 = row['box2d.x1'], row['box2d.y1'], row['box2d.x2'], row['box2d.y2']
            w = x2 - x1
            h = y2 - y1
            cls = CLASS_MAP[row['category']]
            f.write(f"{frame},{obj_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,{cls},1\n")
    return len(sub)

print("Chargement du CSV complet...")
df = pd.read_csv(CSV_PATH, low_memory=False)

for video_name in SELECTED_VIDEOS:
    print(f"\n--- {video_name} ---")
    seq_dir = os.path.join(OUTPUT_ROOT, video_name)
    img_dir = os.path.join(seq_dir, "img1")
    gt_dir = os.path.join(seq_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    n_labels = df[df['videoName'] == video_name]['frameIndex'].nunique()
    video_path = os.path.join(VIDEO_DIR, f"{video_name}.mov")

    n_frames = extract_sampled_frames(video_path, img_dir, n_labels)
    print(f"Frames extraites (échantillonnées) : {n_frames} / attendu {n_labels}")

    gt_path = os.path.join(gt_dir, "gt.txt")
    n_boxes = build_gt(df, video_name, gt_path)
    print(f"Boîtes GT écrites : {n_boxes}")

print("\nTerminé.")
