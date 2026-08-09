
import cv2
import numpy as np
import os
from collections import defaultdict

SEQUENCES = ["0090c713-9d58a186", "012e9465-1031243b", "00ac3256-0f8e2cda"]
MAX_GAP_FRAMES = 20      # fenêtre temporelle max pour chercher une correspondance
MAX_DIST_PX = 80         # distance spatiale max entre disparition et réapparition
HIST_SIM_THRESHOLD = 0.55  # seuil de similarité couleur (0 à 1)

def load_tracks(path):
    tracks = defaultdict(list)  # id -> [(frame, x, y, w, h)]
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            frame, tid = int(p[0]), int(p[1])
            x, y, w, h = map(float, p[2:6])
            tracks[tid].append((frame, x, y, w, h))
    for tid in tracks:
        tracks[tid].sort()
    return tracks

def get_color_hist(img_dir, frame, x, y, w, h):
    img_path = os.path.join(img_dir, f"{frame:06d}.jpg")
    img = cv2.imread(img_path)
    if img is None:
        return None
    H, W = img.shape[:2]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(W, int(x + w)), min(H, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def process_sequence(seq):
    baseline_path = f"results/tracking/baseline/{seq}/track.txt"
    img_dir = f"data/BDD100K_MOT/{seq}/img1"
    tracks = load_tracks(baseline_path)

    # Fin et début de chaque track
    ends = {tid: pts[-1] for tid, pts in tracks.items()}
    starts = {tid: pts[0] for tid, pts in tracks.items()}

    id_remap = {}  # ancien_nouvel_id -> id_correct

    for new_id, (sframe, sx, sy, sw, sh) in starts.items():
        if sframe <= 1:
            continue  # apparu dès la première frame, pas un candidat de switch
        best_match, best_score = None, 0
        for old_id, (eframe, ex, ey, ew, eh) in ends.items():
            if old_id == new_id or eframe >= sframe:
                continue
            gap = sframe - eframe
            if gap > MAX_GAP_FRAMES:
                continue
            dist = np.hypot((sx + sw/2) - (ex + ew/2), (sy + sh/2) - (ey + eh/2))
            if dist > MAX_DIST_PX:
                continue

            h1 = get_color_hist(img_dir, eframe, ex, ey, ew, eh)
            h2 = get_color_hist(img_dir, sframe, sx, sy, sw, sh)
            if h1 is None or h2 is None:
                continue
            sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            if sim > best_score:
                best_score, best_match = sim, old_id

        if best_match is not None and best_score > HIST_SIM_THRESHOLD:
            id_remap[new_id] = best_match

    # Applique le remapping (avec résolution de chaînes A->B->C)
    def resolve(tid):
        seen = set()
        while tid in id_remap and tid not in seen:
            seen.add(tid)
            tid = id_remap[tid]
        return tid

    out_dir = f"results/tracking/notre_hybride/{seq}"
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    with open(baseline_path) as f:
        for line in f:
            p = line.strip().split(",")
            tid = int(p[1])
            new_tid = resolve(tid)
            p[1] = str(new_tid)
            lines.append(",".join(p))
    with open(os.path.join(out_dir, "track.txt"), "w") as f:
        f.write("\n".join(lines))

    print(f"{seq}: {len(id_remap)} switches corrigés sur {len(tracks)} tracks")

for seq in SEQUENCES:
    process_sequence(seq)