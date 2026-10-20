"""Step 1.8: Build the segment table that the RL environment reads.

For every 2-second segment of every stream and every Pareto setting, precompute:
  err      counting error vs the labels (normalised: mean|pred - gt| / (mean gt + 1))
  pred     mean count the system would SEE with that setting (observable at run time)
  refdiff  difference from the most expensive setting (observable at run time, no labels needed)
and the GPU cost of each setting for one segment.

Within a segment, a setting with frame skip k processes every k-th profiled frame and holds the
last count in between. The first frame of every segment is always processed.

Input:  outputs/profiles/profile_counts.parquet, pareto_configs.yaml
Output: outputs/env/segment_table.npz

Run:  python build_segment_table.py
"""
import numpy as np
import pandas as pd
import yaml

from config import ENV_DIR, MIN_EVAL_FRACTION, PROFILE_DIR, PROFILE_STEP, SEGMENT_FRAMES, SEGMENT_SECONDS

PER_SEGMENT = SEGMENT_FRAMES // PROFILE_STEP  # profiled frames per segment (25)


def main():
    pc = pd.read_parquet(PROFILE_DIR / "profile_counts.parquet").sort_values(["stream", "t"]).reset_index(drop=True)
    pareto = sorted(yaml.safe_load(open(PROFILE_DIR / "pareto_configs.yaml")), key=lambda r: r["gpu_ms_per_video_s"])
    names = [r["config"] for r in pareto]
    C = len(names)

    pc["off"] = pc.pos - pc.segment * PER_SEGMENT
    keys = ["stream", "segment"]
    seg_id = pc.groupby(keys, sort=True).ngroup().to_numpy()
    seg = pc.groupby(keys, sort=True).agg(split=("split", "first"), n_frames=("t", "size"),
                                         weather=("weather", lambda w: w.mode().iat[0])).reset_index()
    S = len(seg)

    gt = pc.gt_count.to_numpy(float)
    ev = pc.eval_ok.to_numpy()
    n_all = np.bincount(seg_id, minlength=S)
    n_eval = np.bincount(seg_id, weights=ev.astype(float), minlength=S)
    gt_mean = np.bincount(seg_id, weights=gt * ev, minlength=S) / np.maximum(n_eval, 1)
    valid = n_eval >= MIN_EVAL_FRACTION * n_all

    held = {}
    for r in pareto:
        col = f"{r['model']}_{r['imgsz']}"
        h = pc[col].where(pc.off % r["skip"] == 0).groupby([pc.stream, pc.segment]).ffill()
        held[r["config"]] = h.to_numpy(float)
    ref = held[names[-1]]  # most expensive setting = reference
    ref_mean = np.bincount(seg_id, weights=ref, minlength=S) / n_all

    err = np.full((S, C), np.nan)
    pred = np.zeros((S, C))
    refdiff = np.zeros((S, C))
    for j, name in enumerate(names):
        h = held[name]
        pred[:, j] = np.bincount(seg_id, weights=h, minlength=S) / n_all
        mae = np.bincount(seg_id, weights=np.abs(h - gt) * ev, minlength=S) / np.maximum(n_eval, 1)
        err[:, j] = np.where(valid, mae / (gt_mean + 1), np.nan)
        refdiff[:, j] = (np.bincount(seg_id, weights=np.abs(h - ref), minlength=S) / n_all) / (ref_mean + 1)

    streams = seg.stream.drop_duplicates().tolist()
    stream_of_seg = seg.stream.map({s: i for i, s in enumerate(streams)}).to_numpy()
    stream_start = np.array([np.flatnonzero(stream_of_seg == i)[0] for i in range(len(streams))])
    stream_len = np.bincount(stream_of_seg, minlength=len(streams))
    stream_split = seg.groupby("stream", sort=False).split.first().reindex(streams).to_numpy()

    costs = np.array([r["gpu_ms_per_video_s"] * SEGMENT_SECONDS for r in pareto])  # ms per segment
    err_prior = np.array([r["err_train"] for r in pareto])

    ENV_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ENV_DIR / "segment_table.npz",
        config_names=np.array(names), costs=costs, err_prior=err_prior,
        fps=np.array([r["fps"] for r in pareto]),
        err=err, pred=pred, refdiff=refdiff, gt_mean=gt_mean, valid=valid,
        weather=seg.weather.to_numpy().astype(str), seg_index=seg.segment.to_numpy(),
        stream_of_seg=stream_of_seg, streams=np.array(streams), stream_start=stream_start,
        stream_len=stream_len, stream_split=stream_split.astype(str))

    print(f"{S:,} segments in {len(streams)} streams, {C} settings")
    for split in ("train", "val", "test"):
        m = seg.split.to_numpy() == split
        print(f"  {split:5s}: {m.sum():5d} segments, {valid[m].mean() * 100:5.1f}% valid for scoring")

    # Quick headroom check on the training split: how different is the best setting per segment?
    m = valid & (seg.split.to_numpy() == "train")
    best_idx = np.nanargmin(err[m], axis=1)
    print(f"\nTrain segments where the CHEAPEST setting is already the most accurate: "
          f"{np.mean(best_idx == 0) * 100:.1f}%")
    print("Share of train segments where each setting is the most accurate:")
    for j, name in enumerate(names):
        print(f"  {name:18s} {np.mean(best_idx == j) * 100:5.1f}%   (cost {costs[j]:6.2f} ms per segment)")
    print(f"\nSaved {ENV_DIR / 'segment_table.npz'}")


if __name__ == "__main__":
    main()
