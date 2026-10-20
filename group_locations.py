"""Step 1.4b: Build camera streams and a leakage-safe train/val/test split.

Run AFTER explore.py (it reuses the cached background images).

Two levels:
  STREAM      = one camera. Sequences from the same recording session (consecutive numbers,
                e.g. MVI_40901..MVI_40905) or that look nearly identical are joined.
  SPLIT GROUP = streams that look similar enough that they might be the same road.
                A whole split group always goes to the same split (train, val or test),
                so the same road can never appear in both training and testing.

Optional: outputs/data/manual_merges.yaml lets you force sequences into the same split group:
    - [MVI_20011, MVI_20061]
    - [MVI_40771, MVI_40772]

Outputs (in outputs/data/):
  streams_suggested.yaml       streams + splits (+ split groups, for reference)
  contact_sheet_groups.jpg     backgrounds sorted by split group and stream
  groups_summary.csv           one row per stream

Run:  python group_locations.py
"""
import random

import cv2
import numpy as np
import pandas as pd
import yaml

from config import DATA_OUT, FPS, SEED
from explore import make_contact_sheet, shift_tolerant_similarity, signature

STREAM_SIM = 0.70   # nearly identical view -> same camera stream
GROUP_SIM = 0.55    # similar enough that it might be the same road -> same split
VAL_FRAC = TEST_FRAC = 0.15
N_TRIES = 5000
REQUIRED_WEATHERS = ["night", "rainy", "sunny", "cloudy"]


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)

    def groups(self):
        out = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return sorted((sorted(g) for g in out.values()), key=lambda g: g[0])


def session_of(seq):
    return int(seq.split("_")[1]) // 10   # MVI_40903 -> 4090


def choose_split(group_names, group_frames, group_weathers, total_frames):
    """Try many random splits; keep the one closest to the target sizes with good weather coverage."""
    best, best_score = None, float("inf")
    rng = random.Random(SEED)
    for _ in range(N_TRIES):
        names = group_names[:]
        rng.shuffle(names)
        split = {"test": [], "val": [], "train": []}
        acc = {"test": 0, "val": 0}
        for g in names:
            if acc["test"] < TEST_FRAC * total_frames:
                split["test"].append(g); acc["test"] += group_frames[g]
            elif acc["val"] < VAL_FRAC * total_frames:
                split["val"].append(g); acc["val"] += group_frames[g]
            else:
                split["train"].append(g)
        score = abs(acc["test"] / total_frames - TEST_FRAC) + abs(acc["val"] / total_frames - VAL_FRAC)
        for part in ("val", "test", "train"):
            have = set().union(*[group_weathers[g] for g in split[part]]) if split[part] else set()
            score += 0.5 * sum(w not in have for w in REQUIRED_WEATHERS)
        if score < best_score:
            best, best_score = split, score
    return best


def main():
    seq_df = pd.read_parquet(DATA_OUT / "sequences.parquet")
    seq_df = seq_df[seq_df.has_images]
    seqs = sorted(seq_df.seq)
    weather = seq_df.set_index("seq").weather.to_dict()
    n_frames = seq_df.set_index("seq").n_frames.to_dict()

    bg_dir = DATA_OUT / "backgrounds"
    missing = [s for s in seqs if not (bg_dir / f"{s}.jpg").exists()]
    if missing:
        raise SystemExit(f"Backgrounds missing for {len(missing)} sequences. Run explore.py first.")
    bgs = {s: cv2.imread(str(bg_dir / f"{s}.jpg")) for s in seqs}
    sigs = {s: signature(bgs[s]) for s in seqs}

    n = len(seqs)
    sim = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            sim[i, j] = sim[j, i] = shift_tolerant_similarity(sigs[seqs[i]], sigs[seqs[j]])

    # Level 1: streams.
    uf_stream = UnionFind(seqs)
    for i in range(n):
        for j in range(i + 1, n):
            if session_of(seqs[i]) == session_of(seqs[j]) or sim[i, j] >= STREAM_SIM:
                uf_stream.union(seqs[i], seqs[j])
    streams = {f"s{k + 1:02d}": g for k, g in enumerate(uf_stream.groups())}
    stream_of = {s: name for name, members in streams.items() for s in members}

    # Level 2: split groups (start from streams, then merge similar ones + manual merges).
    uf_group = UnionFind(seqs)
    for members in streams.values():
        for s in members[1:]:
            uf_group.union(members[0], s)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= GROUP_SIM:
                uf_group.union(seqs[i], seqs[j])
    manual_path = DATA_OUT / "manual_merges.yaml"
    if manual_path.exists():
        for merge in yaml.safe_load(open(manual_path)) or []:
            merge = [s for s in merge if s in uf_group.parent]
            for s in merge[1:]:
                uf_group.union(merge[0], s)
        print(f"Applied manual merges from {manual_path.name}")
    seq_groups = uf_group.groups()
    group_of_seq = {s: f"G{k + 1:02d}" for k, g in enumerate(seq_groups) for s in g}
    split_groups = {}
    for name, members in streams.items():
        split_groups.setdefault(group_of_seq[members[0]], []).append(name)

    group_frames = {g: sum(n_frames[s] for st in sts for s in streams[st]) for g, sts in split_groups.items()}
    group_weathers = {g: {weather[s] for st in sts for s in streams[st]} for g, sts in split_groups.items()}
    total = sum(group_frames.values())
    split_by_group = choose_split(sorted(split_groups), group_frames, group_weathers, total)
    splits = {part: sorted(st for g in gs for st in split_groups[g]) for part, gs in split_by_group.items()}
    split_of_stream = {st: part for part, sts in splits.items() for st in sts}

    out = {
        "_note": "Check contact_sheet_groups.jpg. Edit if needed, then save as streams.yaml.",
        "streams": streams,
        "splits": {k: splits[k] for k in ("train", "val", "test")},
        "split_groups": {g: sorted(v) for g, v in sorted(split_groups.items())},
    }
    with open(DATA_OUT / "streams_suggested.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    rows = []
    for name, members in streams.items():
        g = group_of_seq[members[0]]
        rows.append({"stream": name, "split_group": g, "split": split_of_stream[name],
                     "sequences": " ".join(members), "n_seq": len(members),
                     "minutes": round(sum(n_frames[s] for s in members) / FPS / 60, 1),
                     "weathers": ",".join(sorted({weather[s] for s in members}))})
    summary = pd.DataFrame(rows).sort_values(["split", "split_group", "stream"])
    summary.to_csv(DATA_OUT / "groups_summary.csv", index=False)

    order = [s for g in sorted(split_groups) for st in sorted(split_groups[g]) for s in streams[st]]
    labels = {s: f"{group_of_seq[s]} {stream_of[s]}" for s in seqs}
    make_contact_sheet(order, bgs, labels, DATA_OUT / "contact_sheet_groups.jpg")

    print(f"{len(streams)} streams in {len(split_groups)} split groups\n")
    for part in ("train", "val", "test"):
        d = summary[summary.split == part]
        ws = sorted(set(",".join(d.weathers).split(",")))
        print(f"{part:5s}: {len(d):2d} streams, {d.minutes.sum():5.1f} min, weathers: {', '.join(ws)}")
    big = sorted(group_frames.items(), key=lambda kv: -kv[1])[:3]
    print("\nLargest split groups: " + ", ".join(f"{g} ({v / FPS / 60:.1f} min)" for g, v in big))
    print(f"\nSaved streams_suggested.yaml, groups_summary.csv, contact_sheet_groups.jpg to {DATA_OUT}")


if __name__ == "__main__":
    main()