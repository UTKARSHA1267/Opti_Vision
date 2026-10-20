# Phase 1, Part A: Data setup

Goal of this part: turn UA-DETRAC into clean "camera streams" with ground-truth vehicle counts,
ready for profiling (Part B).

## 1. Set up Python (once)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install pandas pyarrow opencv-python pyyaml matplotlib tqdm numpy scipy
```

## 2. Download UA-DETRAC

The original website is often down, so use a mirror (Kaggle or Hugging Face; search "UA-DETRAC").

You need BOTH:
- the **full image sequences** (folders named `MVI_xxxxx` with files `img00001.jpg`, `img00002.jpg`, ...)
- the **XML annotations** (files named `MVI_xxxxx.xml`, train and test)

Avoid versions that were converted to YOLO format or shuffled into a single image folder:
they lose the video order and the ignored regions, which this project needs.

Put everything under one folder, e.g. `datasets/DETRAC/`. The exact layout inside does not
matter; the scripts search for it. Otherwise set the path:

```bash
export DETRAC_DIR=/path/to/DETRAC      # Windows: set DETRAC_DIR=C:\path\to\DETRAC
```

## 3. Run the scripts in order

| Step | Command | What it does | Check |
|---|---|---|---|
| 1 | `python counting.py` | Tests the counting rule | Prints "self-test passed" |
| 2 | `python parse_detrac.py` | Reads all XML files into tables | About 100 sequences and about 140,000 frames |
| 3 | `python explore.py` | Stats, contact sheet, suggested location groups | Open `outputs/data/contact_sheet.jpg` |
| 4 | Manual | Fix the groups (see below) | Save as `outputs/data/streams.yaml` |
| 5 | `python build_streams.py` | Stitches sequences into camera streams | Read the summary table |

## 4. Fixing the location groups (important)

UA-DETRAC was recorded at 24 locations, but the files don't say which sequence is where.
`explore.py` guesses by comparing first frames. It makes mistakes, especially between day
and night recordings of the same place.

1. Open `contact_sheet.jpg`. Each thumbnail is labelled `MVI_xxxxx [locNN]`.
2. Open `streams_suggested.yaml`.
3. Move sequences so each group contains only views of the same road from the same camera angle.
   Merge groups that are the same place; split groups that mix places.
4. Check the `splits` section: about 70% of groups in `train`, 15% in `val`, 15% in `test`.
   Try to have night and rainy streams in `val` and `test` too.
   A location must never appear in two splits.
5. Save the file as `streams.yaml` in the same folder.

Use `nearest_neighbours.csv` to help: it lists the 3 most similar sequences for each one.
The thumbnails show 'empty road' backgrounds (median of 15 frames), so vehicles are removed.

## 5. Files produced

| File | Contents |
|---|---|
| `sequences.parquet` | One row per sequence: weather, camera state, length |
| `frames.parquet` | One row per frame: image path, ground-truth count |
| `boxes.parquet` | Every annotated vehicle box |
| `ignored_regions.json` | Areas where vehicles are not counted |
| `stream_index.parquet` | Every frame of every stream with its 2-second segment number and split |
| `streams_summary.csv` | Length, weather and average traffic of each stream |

## Counting rule (used everywhere)

A vehicle counts if the centre of its box is inside the frame and not inside an ignored region.
YOLO predictions will use exactly the same rule (`counting.py`), so ground truth and
predictions are always comparable. Frames flagged `annotated=False` or `near_boundary=True`
are excluded from evaluation later.

## Done when

- [ ] `parse_detrac.py` reports about 100 sequences and about 140,000 frames
- [ ] Groups checked by eye on the contact sheet
- [ ] `build_streams.py` runs, every split has some streams, test includes night or rain