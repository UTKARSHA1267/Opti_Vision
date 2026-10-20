"""Step 1.3: Parse UA-DETRAC XML annotations into tidy tables.

Outputs (in outputs/data/):
  sequences.parquet  one row per video sequence (weather, camera state, length, ...)
  frames.parquet     one row per frame (image path, ground-truth vehicle count, ...)
  boxes.parquet      one row per annotated vehicle box
  ignored_regions.json   ignored regions per sequence (used by the counting rule)

Run:  python parse_detrac.py
"""
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from config import DATA_OUT, DETRAC_DIR, FRAME_H, FRAME_W
from counting import count_vehicles

SEQ_RE = re.compile(r"^MVI_\d+$")
IMG_RE = re.compile(r"^img(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def find_files(root):
    """Walk the dataset folder once; return {seq: xml_path} and {seq: image_dir}."""
    xmls, img_dirs = {}, {}
    for dirpath, _, filenames in os.walk(root):
        base = os.path.basename(dirpath)
        if SEQ_RE.match(base) and any(IMG_RE.match(f) for f in filenames):
            img_dirs[base] = dirpath
        for f in filenames:
            stem, ext = os.path.splitext(f)
            if ext.lower() == ".xml" and SEQ_RE.match(stem):
                xmls[stem] = os.path.join(dirpath, f)
    return xmls, img_dirs


def origin_of(path):
    p = path.lower()
    if "train" in p:
        return "train"
    if "test" in p:
        return "test"
    return "unknown"


def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    attr = root.find("sequence_attribute")
    attr = attr.attrib if attr is not None else {}
    # The dataset spells it 'sence_weather'; accept the correct spelling too.
    weather = attr.get("sence_weather", attr.get("scene_weather", "unknown"))
    camera_state = attr.get("camera_state", "unknown")

    ignored = []
    ign = root.find("ignored_region")
    if ign is not None:
        for b in ign.findall("box"):
            ignored.append([float(b.get(k)) for k in ("left", "top", "width", "height")])

    frames = {}
    boxes = []
    for fr in root.findall("frame"):
        num = int(fr.get("num"))
        density = int(fr.get("density", 0))
        fboxes = []
        tl = fr.find("target_list")
        if tl is not None:
            for tgt in tl.findall("target"):
                b = tgt.find("box")
                a = tgt.find("attribute")
                a = a.attrib if a is not None else {}
                box = [float(b.get(k)) for k in ("left", "top", "width", "height")]
                fboxes.append(box)
                boxes.append({
                    "frame": num,
                    "target_id": int(tgt.get("id")),
                    "left": box[0], "top": box[1], "width": box[2], "height": box[3],
                    "vehicle_type": a.get("vehicle_type", "unknown"),
                    "truncation_ratio": float(a.get("truncation_ratio", 0) or 0),
                })
        frames[num] = {"density": density, "boxes": fboxes}
    return weather, camera_state, ignored, frames, boxes


def main():
    if not DETRAC_DIR.exists():
        raise SystemExit(f"Dataset folder not found: {DETRAC_DIR}\n"
                         "Set DETRAC_DIR in config.py or as an environment variable.")
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Searching {DETRAC_DIR} ...")
    xmls, img_dirs = find_files(DETRAC_DIR)
    print(f"Found {len(xmls)} XML files and {len(img_dirs)} image folders")
    if not xmls:
        raise SystemExit("No MVI_*.xml annotation files found. Check the download includes the XML annotations.")

    seq_rows, frame_rows, box_rows = [], [], []
    ignored_all = {}

    for seq, xml_path in tqdm(sorted(xmls.items()), desc="Parsing"):
        weather, camera_state, ignored, frames, boxes = parse_xml(xml_path)
        ignored_all[seq] = ignored

        img_dir = img_dirs.get(seq)
        img_files = {}
        if img_dir:
            for f in os.listdir(img_dir):
                m = IMG_RE.match(f)
                if m:
                    img_files[int(m.group(1))] = os.path.join(img_dir, f)

        n_frames = max(img_files) if img_files else (max(frames) if frames else 0)
        missing_annot = 0
        for num in range(1, n_frames + 1):
            info = frames.get(num)
            if info is None:
                missing_annot += 1
            fboxes = info["boxes"] if info else []
            frame_rows.append({
                "seq": seq,
                "frame": num,
                "img_path": img_files.get(num, ""),
                "gt_count": count_vehicles(fboxes, ignored, FRAME_W, FRAME_H),
                "n_boxes_total": len(fboxes),
                "density": info["density"] if info else 0,
                "annotated": info is not None,
            })
        for b in boxes:
            b["seq"] = seq
        box_rows.extend(boxes)

        seq_rows.append({
            "seq": seq,
            "origin": origin_of(xml_path),
            "weather": weather,
            "camera_state": camera_state,
            "n_frames": n_frames,
            "minutes": round(n_frames / 25 / 60, 2),
            "n_ignored_regions": len(ignored),
            "has_images": bool(img_files),
            "missing_annotation_frames": missing_annot,
        })

    seq_df = pd.DataFrame(seq_rows)
    frame_df = pd.DataFrame(frame_rows)
    box_df = pd.DataFrame(box_rows)

    seq_df.to_parquet(DATA_OUT / "sequences.parquet", index=False)
    frame_df.to_parquet(DATA_OUT / "frames.parquet", index=False)
    box_df.to_parquet(DATA_OUT / "boxes.parquet", index=False)
    with open(DATA_OUT / "ignored_regions.json", "w") as f:
        json.dump(ignored_all, f)

    print("\n=== Summary ===")
    print(f"Sequences: {len(seq_df)}  (with images: {int(seq_df.has_images.sum())})")
    print(f"Frames: {len(frame_df):,}  ({len(frame_df) / 25 / 3600:.2f} hours of video)")
    print(f"Boxes: {len(box_df):,}")
    print("\nSequences per weather:\n" + seq_df.weather.value_counts().to_string())
    no_img = seq_df[~seq_df.has_images].seq.tolist()
    if no_img:
        print(f"\nWARNING: {len(no_img)} sequences have annotations but no images: {no_img[:10]} ...")
    gaps = seq_df[seq_df.missing_annotation_frames > 0]
    if len(gaps):
        print(f"\nNote: {len(gaps)} sequences have frames with no annotation entry "
              f"(counted as 0, flagged annotated=False). Exclude those frames from evaluation.")
    print(f"\nSaved to {DATA_OUT}")


if __name__ == "__main__":
    main()
