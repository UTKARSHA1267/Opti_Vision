"""Step 1.5: Stitch sequences from the same location into long 'camera streams'.

Input:   outputs/data/streams.yaml  (copy of streams_suggested.yaml after you checked it)
Output:  outputs/data/stream_index.parquet  one row per frame of every stream, with:
           stream, split, t (frame index in stream), segment (2-second decision step),
           seq, frame, img_path, gt_count, weather, near_boundary, annotated
         outputs/data/streams_summary.csv

Run:  python build_streams.py
"""
import pandas as pd
import yaml

from config import BOUNDARY_MARGIN_FRAMES, DATA_OUT, FPS, SEGMENT_FRAMES


def main():
    cfg_path = DATA_OUT / "streams.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"{cfg_path} not found. Check streams_suggested.yaml, fix it, save it as streams.yaml.")
    cfg = yaml.safe_load(open(cfg_path))
    streams, splits = cfg["streams"], cfg["splits"]

    split_of = {s: name for name, members in splits.items() for s in members}
    missing = [s for s in streams if s not in split_of]
    if missing:
        raise SystemExit(f"These streams are not in any split: {missing}")
    seq_lists = [seq for members in streams.values() for seq in members]
    dupes = {s for s in seq_lists if seq_lists.count(s) > 1}
    if dupes:
        raise SystemExit(f"Sequences used in more than one stream: {sorted(dupes)}")

    frames = pd.read_parquet(DATA_OUT / "frames.parquet")
    seq_info = pd.read_parquet(DATA_OUT / "sequences.parquet").set_index("seq")

    parts = []
    for stream, members in streams.items():
        offset, joins, stream_parts = 0, [], []
        for seq in members:
            d = frames[(frames.seq == seq) & (frames.img_path != "")].sort_values("frame").copy()
            if d.empty:
                print(f"WARNING: {seq} has no images, skipped")
                continue
            if offset > 0:
                joins.append(offset)
            d["t"] = range(offset, offset + len(d))
            d["weather"] = seq_info.loc[seq, "weather"]
            offset += len(d)
            stream_parts.append(d)
        if not stream_parts:
            continue
        s = pd.concat(stream_parts, ignore_index=True)
        # Mark frames close to a join between two sequences (sudden scene jumps).
        near = pd.Series(False, index=s.index)
        for j in joins:
            near |= (s.t >= j - BOUNDARY_MARGIN_FRAMES) & (s.t < j + BOUNDARY_MARGIN_FRAMES)
        s["near_boundary"] = near
        parts.append(s.assign(stream=stream, split=split_of[stream]))
    idx = pd.concat(parts, ignore_index=True)
    idx["segment"] = idx.t // SEGMENT_FRAMES

    cols = ["stream", "split", "t", "segment", "seq", "frame", "img_path",
            "gt_count", "weather", "near_boundary", "annotated"]
    idx = idx[cols].sort_values(["stream", "t"]).reset_index(drop=True)
    idx.to_parquet(DATA_OUT / "stream_index.parquet", index=False)

    summary = idx.groupby(["stream", "split"]).agg(
        sequences=("seq", "nunique"),
        frames=("t", "size"),
        segments=("segment", "nunique"),
        mean_vehicles=("gt_count", "mean"),
        weathers=("weather", lambda w: ",".join(sorted(set(w)))),
    ).reset_index()
    summary["minutes"] = (summary.frames / FPS / 60).round(1)
    summary["mean_vehicles"] = summary.mean_vehicles.round(1)
    summary.to_csv(DATA_OUT / "streams_summary.csv", index=False)

    print(summary.to_string(index=False))
    print("\nMinutes of video per split:")
    print((idx.groupby("split").size() / FPS / 60).round(1).to_string())
    print(f"\nFrames near a join (excluded from evaluation later): {int(idx.near_boundary.sum()):,}")
    print(f"Saved {DATA_OUT / 'stream_index.parquet'}")


if __name__ == "__main__":
    main()
