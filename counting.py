"""Counting rules, used for BOTH ground truth and YOLO predictions so they are always comparable.

Rule: a vehicle is counted if the centre of its box is inside the frame and NOT inside
any 'ignored region'. UA-DETRAC leaves vehicles inside ignored regions unannotated
(e.g. far-away parking areas), so predictions there must be dropped too.
"""
import numpy as np


def box_centers(boxes: np.ndarray) -> np.ndarray:
    """boxes: (N, 4) as [left, top, width, height] -> (N, 2) centre points."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.stack([boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], axis=1)


def points_in_rects(points: np.ndarray, rects: np.ndarray) -> np.ndarray:
    """True for each point that lies inside at least one rect ([left, top, width, height])."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    rects = np.asarray(rects, dtype=float).reshape(-1, 4)
    if len(points) == 0 or len(rects) == 0:
        return np.zeros(len(points), dtype=bool)
    x, y = points[:, 0:1], points[:, 1:2]
    l, t = rects[:, 0], rects[:, 1]
    r, b = l + rects[:, 2], t + rects[:, 3]
    inside = (x >= l) & (x <= r) & (y >= t) & (y <= b)
    return inside.any(axis=1)


def keep_mask(boxes, ignored_regions, frame_w: int, frame_h: int) -> np.ndarray:
    """Which boxes count: centre inside the frame and outside every ignored region."""
    c = box_centers(boxes)
    if len(c) == 0:
        return np.zeros(0, dtype=bool)
    in_frame = (c[:, 0] >= 0) & (c[:, 0] < frame_w) & (c[:, 1] >= 0) & (c[:, 1] < frame_h)
    return in_frame & ~points_in_rects(c, ignored_regions)


def count_vehicles(boxes, ignored_regions, frame_w: int, frame_h: int) -> int:
    return int(keep_mask(boxes, ignored_regions, frame_w, frame_h).sum())


def dedupe_detections(df, iou_thr=0.7):
    """Remove duplicate boxes on the same vehicle ACROSS classes (e.g. one taxi detected as both
    'car' and 'truck'). Within each frame, a box is dropped if a MORE confident box overlaps it by
    IoU >= iou_thr. Boxes are only compared with boxes in the same frame, so this stays fast.
    (Equivalent to standard NMS except for rare chains of three overlapping boxes.)
    df needs columns: seq, frame, left, top, width, height, conf."""
    if len(df) == 0:
        return df
    df = df.reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)
    cols = ["frame", "left", "top", "width", "height", "conf"]
    for _, g in df.groupby("seq", sort=False):
        a = g[cols].assign(_i=g.index.to_numpy())
        p = a.merge(a, on="frame", suffixes=("_a", "_b"))
        stronger = (p.conf_a > p.conf_b) | ((p.conf_a == p.conf_b) & (p._i_a < p._i_b))
        p = p[stronger]
        if p.empty:
            continue
        ix = np.clip(np.minimum(p.left_a + p.width_a, p.left_b + p.width_b) - np.maximum(p.left_a, p.left_b), 0, None)
        iy = np.clip(np.minimum(p.top_a + p.height_a, p.top_b + p.height_b) - np.maximum(p.top_a, p.top_b), 0, None)
        inter = ix * iy
        union = p.width_a * p.height_a + p.width_b * p.height_b - inter
        iou = inter / np.maximum(union, 1e-6)
        keep[p._i_b[iou >= iou_thr].unique()] = False
    return df[keep].reset_index(drop=True)


if __name__ == "__main__":
    # Tiny self-test.
    boxes = [[10, 10, 20, 20], [500, 300, 40, 40], [900, 20, 30, 30]]
    ignored = [[850, 0, 110, 100]]  # covers the third box
    assert count_vehicles(boxes, ignored, 960, 540) == 2
    assert count_vehicles([], ignored, 960, 540) == 0
    import pandas as pd
    d = pd.DataFrame({"seq": ["A"] * 3 + ["B"], "frame": [1, 1, 1, 1],
                      "left": [100, 102, 400, 100], "top": [100, 101, 100, 100],
                      "width": [50, 50, 50, 50], "height": [40, 40, 40, 40], "conf": [0.9, 0.6, 0.8, 0.7]})
    out = dedupe_detections(d)
    assert len(out) == 3 and 0.6 not in out.conf.tolist(), out  # duplicate in A removed, B untouched
    print("counting.py self-test passed")