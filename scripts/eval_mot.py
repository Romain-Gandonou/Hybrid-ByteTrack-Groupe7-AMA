import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm
import pandas as pd

SEQUENCES = ["0090c713-9d58a186", "012e9465-1031243b", "00ac3256-0f8e2cda"]
TAGS = ["baseline", "hybride", "notre_hybride"]

def load(path):
    df = pd.read_csv(path, header=None, usecols=[0,1,2,3,4,5],
                      names=["frame","id","x","y","w","h"])
    return df

results = []
for tag in TAGS:
    acc = mm.MOTAccumulator(auto_id=True)
    for seq in SEQUENCES:
        gt = load(f"data/BDD100K_MOT/{seq}/gt/gt.txt")
        hyp = load(f"results/tracking/{tag}/{seq}/track.txt")
        for frame in sorted(gt["frame"].unique()):
            g = gt[gt["frame"] == frame]
            h = hyp[hyp["frame"] == frame]
            g_ids = g["id"].tolist()
            h_ids = h["id"].tolist()
            g_boxes = g[["x","y","w","h"]].values
            h_boxes = h[["x","y","w","h"]].values
            dists = mm.distances.iou_matrix(g_boxes, h_boxes, max_iou=0.5)
            acc.update(g_ids, h_ids, dists)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota","idf1","num_switches","num_false_positives","num_misses"], name=tag)
    results.append(summary)

final = pd.concat(results)
print(final)
final.to_csv("results/comparison_mota_idf1.csv")