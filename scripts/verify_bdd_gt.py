
import cv2
import os

def draw_gt_on_frame(gt_txt, img_dir, frame_number, output_path):
    img_path = os.path.join(img_dir, f"{frame_number:06d}.jpg")
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Erreur: impossible de lire {img_path}")
        return

    CLASS_NAMES = {1: "car", 2: "moto"}
    CLASS_COLORS = {1: (255, 0, 0), 2: (0, 0, 255)}  # bleu=voiture, rouge=moto

    with open(gt_txt) as f:
        for line in f:
            parts = line.strip().split(',')
            fid = int(parts[0])
            if fid != frame_number:
                continue
            obj_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            cls = int(parts[7])
            x, y, w, h = int(x), int(y), int(w), int(h)
            color = CLASS_COLORS.get(cls, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{CLASS_NAMES.get(cls, '?')} #{obj_id}"
            cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, frame)
    print(f"Sauvegardé -> {output_path}")

os.makedirs("results", exist_ok=True)

SEQUENCES = ["0090c713-9d58a186", "012e9465-1031243b", "00ac3256-0f8e2cda"]
for seq in SEQUENCES:
    draw_gt_on_frame(
        f"data/BDD100K_MOT/{seq}/gt/gt.txt",
        f"data/BDD100K_MOT/{seq}/img1",
        50,
        f"results/check_bdd_{seq}_frame50.jpg"
    )