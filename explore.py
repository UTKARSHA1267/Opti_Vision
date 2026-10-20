"""Step 1.4: Explore the data and suggest which sequences come from the same camera location.

Outputs (in outputs/data/):
  contact_sheet.jpg        first frame of every sequence, labelled -> look at it to check groups
  count_timelines.png      ground-truth vehicle count over time for a few sequences
  streams_suggested.yaml   automatic guess of location groups + a train/val/test split

UA-DETRAC was recorded at 24 locations, but the files don't say which sequence is
where. We guess by comparing how similar the first frames look, then YOU check the
contact sheet and fix mistakes before building streams.

Run:  python explore.py
"""
'''
import random

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import DATA_OUT, SEED  # noqa: E402

THUMB_W, THUMB_H, COLS = 240, 135, 8
SIMILARITY_THRESHOLD = 0.85  # raise it if unrelated scenes get grouped; lower it if same scenes split


def first_frame_path(frames, seq):
    rows = frames[(frames.seq == seq) & (frames.img_path != "")]
    return rows.sort_values("frame").img_path.iloc[0] if len(rows) else None


def make_contact_sheet(seqs, paths, labels, out_path):
    rows = (len(seqs) + COLS - 1) // COLS
    sheet = np.full((rows * THUMB_H, COLS * THUMB_W, 3), 30, np.uint8)
    for i, seq in enumerate(seqs):
        img = cv2.imread(paths[seq]) if paths.get(seq) else None
        if img is None:
            continue
        thumb = cv2.resize(img, (THUMB_W, THUMB_H))
        text = f"{seq} [{labels.get(seq, '?')}]"
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 22), (0, 0, 0), -1)
        cv2.putText(thumb, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        r, c = divmod(i, COLS)
        sheet[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = thumb
    cv2.imwrite(str(out_path), sheet)


def signature(path):
    """Small, normalised edge image: captures the road layout, mostly ignores lighting."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (96, 54))
    img = cv2.equalizeHist(img)  # reduces day/night brightness differences
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 150).astype(np.float32)
    v = edges.flatten()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def suggest_groups(seqs, paths):
    """Group sequences whose first frames look alike (union-find on a similarity threshold)."""
    valid = [s for s in seqs if paths.get(s)]
    sigs = np.stack([signature(paths[s]) for s in valid])
    sim = sigs @ sigs.T  # cosine similarity between every pair of first frames
    parent = list(range(len(valid)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if sim[i, j] >= SIMILARITY_THRESHOLD:
                parent[find(i)] = find(j)

    groups = {}
    for i, s in enumerate(valid):
        groups.setdefault(find(i), []).append(s)
    ordered = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    return {f"loc{k + 1:02d}": g for k, g in enumerate(ordered)}


def suggest_split(stream_names):
    names = list(stream_names)
    random.Random(SEED).shuffle(names)
    n = len(names)
    n_test = max(1, round(0.15 * n))
    n_val = max(1, round(0.15 * n))
    return {
        "train": sorted(names[n_test + n_val:]),
        "val": sorted(names[n_test:n_test + n_val]),
        "test": sorted(names[:n_test]),
    }


def plot_timelines(frames, seqs, out_path):
    pick = seqs[:: max(1, len(seqs) // 6)][:6]
    fig, axes = plt.subplots(len(pick), 1, figsize=(10, 2 * len(pick)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, seq in zip(axes, pick):
        d = frames[frames.seq == seq]
        ax.plot(d.frame / 25, d.gt_count, lw=0.8)
        ax.set_title(seq, fontsize=9)
        ax.set_ylabel("vehicles")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    seq_df = pd.read_parquet(DATA_OUT / "sequences.parquet")
    frames = pd.read_parquet(DATA_OUT / "frames.parquet")
    seqs = sorted(seq_df[seq_df.has_images].seq)

    print("=== Vehicles per frame (ground truth, after ignored regions) ===")
    print(frames.gt_count.describe().round(2).to_string())
    per_w = frames.merge(seq_df[["seq", "weather"]], on="seq").groupby("weather").gt_count.mean()
    print("\nMean vehicles per frame by weather:\n" + per_w.round(2).to_string())

    paths = {s: first_frame_path(frames, s) for s in seqs}
    groups = suggest_groups(seqs, paths)
    label_of = {s: name for name, members in groups.items() for s in members}

    make_contact_sheet(seqs, paths, label_of, DATA_OUT / "contact_sheet.jpg")
    plot_timelines(frames, seqs, DATA_OUT / "count_timelines.png")

    suggestion = {
        "_note": "Auto-suggested. Check contact_sheet.jpg, fix groups, then save as streams.yaml.",
        "streams": groups,
        "splits": suggest_split(groups.keys()),
    }
    with open(DATA_OUT / "streams_suggested.yaml", "w") as f:
        yaml.safe_dump(suggestion, f, sort_keys=False)

    sizes = sorted((len(g) for g in groups.values()), reverse=True)
    print(f"\nSuggested {len(groups)} location groups (UA-DETRAC has 24 locations). Group sizes: {sizes}")
    if len(groups) > 35:
        print("Many tiny groups -> try lowering SIMILARITY_THRESHOLD.")
    elif len(groups) < 15:
        print("Few large groups -> try raising SIMILARITY_THRESHOLD.")
    print(f"Saved contact sheet, timelines and streams_suggested.yaml to {DATA_OUT}")


if __name__ == "__main__":
    main()
'''
"""Step 1.4: Explore the data and suggest which sequences come from the same camera location.

Outputs (in outputs/data/):
  backgrounds/             one 'empty road' image per sequence (median of several frames)
  contact_sheet.jpg        all backgrounds, SORTED BY SUGGESTED GROUP -> check this by eye
  nearest_neighbours.csv   for each sequence, its 3 most similar sequences and their scores
  count_timelines.png      ground-truth vehicle count over time for a few sequences
  streams_suggested.yaml   suggested location groups + a train/val/test split

UA-DETRAC was recorded at 24 locations, but the files don't say which sequence is where.
We build an empty-road background per sequence, compare their shapes (edges) while
allowing small camera shifts, and cluster them into N_LOCATIONS groups.
YOU then check the contact sheet and fix mistakes before building streams.

Run:  python explore.py
"""
import random

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import DATA_OUT, SEED  # noqa: E402

N_LOCATIONS = 24          # UA-DETRAC was recorded at 24 locations
N_BG_FRAMES = 15          # frames used to build each background
THUMB_W, THUMB_H, COLS = 240, 135, 8
SIG_W, SIG_H = 96, 54     # size used for comparison
MAX_SHIFT = 6             # pixels of camera shift tolerated (at comparison size)


def background(paths):
    """Median of several frames: moving vehicles disappear, the road stays."""
    idx = np.linspace(0, len(paths) - 1, min(N_BG_FRAMES, len(paths))).astype(int)
    imgs = [cv2.imread(paths[i]) for i in idx]
    imgs = [im for im in imgs if im is not None]
    return np.median(np.stack(imgs), axis=0).astype(np.uint8)


def signature(bg):
    """Lighting-robust shape image: equalised grayscale -> gradient magnitude -> blurred."""
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)
    g = cv2.equalizeHist(g)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.GaussianBlur(mag, (5, 5), 0)
    return mag / (mag.max() + 1e-6)


def shift_tolerant_similarity(a, b):
    """Best normalised correlation of a's centre inside b (and vice versa): tolerates small shifts."""
    m = MAX_SHIFT
    s1 = cv2.matchTemplate(b, a[m:-m, m:-m], cv2.TM_CCOEFF_NORMED).max()
    s2 = cv2.matchTemplate(a, b[m:-m, m:-m], cv2.TM_CCOEFF_NORMED).max()
    return float((s1 + s2) / 2)


def make_contact_sheet(order, bgs, labels, out_path):
    rows = (len(order) + COLS - 1) // COLS
    sheet = np.full((rows * THUMB_H, COLS * THUMB_W, 3), 30, np.uint8)
    for i, seq in enumerate(order):
        thumb = cv2.resize(bgs[seq], (THUMB_W, THUMB_H))
        cv2.rectangle(thumb, (0, 0), (THUMB_W, 22), (0, 0, 0), -1)
        cv2.putText(thumb, f"{labels[seq]}  {seq}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        r, c = divmod(i, COLS)
        sheet[r * THUMB_H:(r + 1) * THUMB_H, c * THUMB_W:(c + 1) * THUMB_W] = thumb
    cv2.imwrite(str(out_path), sheet)


def suggest_split(stream_names):
    names = list(stream_names)
    random.Random(SEED).shuffle(names)
    n = len(names)
    n_test = max(1, round(0.15 * n))
    n_val = max(1, round(0.15 * n))
    return {
        "train": sorted(names[n_test + n_val:]),
        "val": sorted(names[n_test:n_test + n_val]),
        "test": sorted(names[:n_test]),
    }


def plot_timelines(frames, seqs, out_path):
    pick = seqs[:: max(1, len(seqs) // 6)][:6]
    fig, axes = plt.subplots(len(pick), 1, figsize=(10, 2 * len(pick)))
    axes = np.atleast_1d(axes)
    for ax, seq in zip(axes, pick):
        d = frames[frames.seq == seq]
        ax.plot(d.frame / 25, d.gt_count, lw=0.8)
        ax.set_title(seq, fontsize=9)
        ax.set_ylabel("vehicles")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    seq_df = pd.read_parquet(DATA_OUT / "sequences.parquet")
    frames = pd.read_parquet(DATA_OUT / "frames.parquet")
    seqs = sorted(seq_df[seq_df.has_images].seq)
    weather = seq_df.set_index("seq").weather.to_dict()

    print("=== Vehicles per frame (ground truth, after ignored regions) ===")
    print(frames.gt_count.describe().round(2).to_string())
    per_w = frames.merge(seq_df[["seq", "weather"]], on="seq").groupby("weather").gt_count.mean()
    print("\nMean vehicles per frame by weather:\n" + per_w.round(2).to_string())

    # 1. Background image per sequence (cached, so re-runs are fast).
    bg_dir = DATA_OUT / "backgrounds"
    bg_dir.mkdir(exist_ok=True)
    bgs = {}
    for seq in tqdm(seqs, desc="Backgrounds"):
        p = bg_dir / f"{seq}.jpg"
        if p.exists():
            bgs[seq] = cv2.imread(str(p))
        else:
            paths = frames[(frames.seq == seq) & (frames.img_path != "")].sort_values("frame").img_path.tolist()
            bgs[seq] = background(paths)
            cv2.imwrite(str(p), bgs[seq])

    # 2. Pairwise similarity.
    sigs = {s: signature(bgs[s]) for s in seqs}
    n = len(seqs)
    sim = np.eye(n)
    for i in tqdm(range(n), desc="Comparing"):
        for j in range(i + 1, n):
            sim[i, j] = sim[j, i] = shift_tolerant_similarity(sigs[seqs[i]], sigs[seqs[j]])

    # 3. Cluster into N_LOCATIONS groups (average linkage on distance = 1 - similarity).
    dist = np.clip(1 - sim, 0, None)
    np.fill_diagonal(dist, 0)
    k = min(N_LOCATIONS, n)
    labels_num = fcluster(linkage(squareform(dist, checks=False), method="average"), k, criterion="maxclust")
    groups = {}
    for s, lab in zip(seqs, labels_num):
        groups.setdefault(lab, []).append(s)
    ordered = sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])
    streams = {f"loc{i + 1:02d}": g for i, g in enumerate(ordered)}
    label_of = {s: name for name, members in streams.items() for s in members}

    # 4. Nearest neighbours table (helps when fixing groups by hand).
    rows = []
    sim_nn = sim.copy()
    np.fill_diagonal(sim_nn, -np.inf)  # never list a sequence as its own neighbour
    for i, s in enumerate(seqs):
        nn = np.argsort(-sim_nn[i])[:3]
        rows.append({"seq": s, "group": label_of[s], "weather": weather.get(s, "?"),
                     **{f"nn{r + 1}": f"{seqs[j]} ({sim[i, j]:.2f})" for r, j in enumerate(nn)}})
    pd.DataFrame(rows).to_csv(DATA_OUT / "nearest_neighbours.csv", index=False)

    # 5. Contact sheet sorted by group, timelines, suggested yaml.
    order = [s for members in streams.values() for s in members]
    make_contact_sheet(order, bgs, label_of, DATA_OUT / "contact_sheet.jpg")
    plot_timelines(frames, seqs, DATA_OUT / "count_timelines.png")
    suggestion = {
        "_note": "Auto-suggested. Check contact_sheet.jpg, fix groups, then save as streams.yaml.",
        "streams": streams,
        "splits": suggest_split(streams.keys()),
    }
    with open(DATA_OUT / "streams_suggested.yaml", "w") as f:
        yaml.safe_dump(suggestion, f, sort_keys=False)

    sizes = sorted((len(g) for g in streams.values()), reverse=True)
    print(f"\nSuggested {len(streams)} location groups. Group sizes: {sizes}")
    print("Open contact_sheet.jpg: thumbnails are sorted by group, so each group sits together.")
    print(f"Saved to {DATA_OUT}")


if __name__ == "__main__":
    main()