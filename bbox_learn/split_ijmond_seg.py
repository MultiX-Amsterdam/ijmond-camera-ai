"""
This script splits the IJMOND segmentation dataset into training, validation, and test sets.
"""

"""
First, we need to read the metadata file that contains the cropped images and masks information.
The cropped images with masks are in the dataset/ijmond_seg/test/cropped/test_with_mask.txt file
The cropped images without masks are in the dataset/ijmond_seg/test/cropped/test_without_mask.txt file
Each line in the file looks like below:
images_npy/kooks_1__2024-10-31T06-51-46Z_frame_2396_jpg.rf.dd0990aa1ac67e5bfaa9ba7fe7b6f8b1_crop_2.npy masks_npy/kooks_1__2024-10-31T06-51-46Z_frame_2396_jpg.rf.dd0990aa1ac67e5bfaa9ba7fe7b6f8b1_crop_2.npy
If we seperate by space, the first part is the image path and the second part is the mask path.
In this way, we can get a list of file names of image-mask pairs for the dataset.

Next, we need to get all the timestamps and camera views from the file names.
The timestamp is the part after the second double underscore and before the "_frame" part.
The camera view is the part before the first double underscore.
For example, in the file name "kooks_1__2024-10-31T06-51-46Z_frame_2396_jpg.rf.dd0990aa1ac67e5bfaa9ba7fe7b6f8b1_crop_2.npy",
the timestamp is "2024-10-31T06-51-46Z" and the camera view is "kooks_1".
We have three different camera views: "kooks_1", "kooks_2", and "hoogovens_6_7".
So, for each file name, we can extract the timestamp and camera view.
We also need to convert the timestamp to a datetime object for easier comparison.
After that, print a distribution of how many images for each day we have in the dataset.
Also print the number of images for each camera view.
Do the printing for both the images with masks only, and the images without masks.
"""

"""
Now, we can split the dataset into training, validation, and test sets based on the timestamps.
First, we need to merge the images with masks and without masks into one list.
Then, we can sort the merged list by timestamps.
We need to keep track of which images have masks and which do not.
There will be two different types of splits:
- 1. Split by timestamp only: training set first 70% sorted by timestamps, validation set the next 10%, test set the final 20%
- 2. Split by camera view only: training set the first 80% of "kooks_2" sorted by timestamps, validation set the rest of 20% of "kooks_2" sorted by timestamps, test set "hoogovens_6_7" and "kooks_1"
Now there is one more thing: for the training split, we need to create five different versions of it, with 100%, 80%, 60%, 40%, and 20% of the training data, sorted by timestamps.
The 100% training split is the full training set.
The 80% training split is the last 80% of the training set sorted by timestamps.
The 60% training split is the last 60% of the training set sorted by timestamps.
The 40% training split is the last 40% of the training set sorted by timestamps.
The 20% training split is the last 20% of the training set sorted by timestamps.
The reason for this is to simulate different amounts of data available for training.
These percentages are the "last" part in the training set according to sorted timestamps to ensure that the timestamps, when considered together with the validation and test sets, are continuous.
Now, we can proceed to split the dataset into the following structure:
  - train/100
  - train/80
  - train/60
  - train/40
  - train/20
  - val
  - test
Next, we need to seperate each split into two files: one with masks and one without masks.
For example, for the training split with 100% data, we will have two files:
- train/100_with_masks.txt
- train/100_without_masks.txt
The final output should look like below:
- the splits data should be saved in "dataset/ijmond_seg/test/cropped/splits/"
- it should contain two types of splits in two subfolders: "split_by_timestamp/" and "split_by_camera/"
- each split in a subfolder should contain the following files:
  - train/100_with_masks.txt
  - train/100_without_masks.txt
  - train/80_with_masks.txt
  - train/80_without_masks.txt
  - train/60_with_masks.txt
  - train/60_without_masks.txt
  - train/40_with_masks.txt
  - train/40_without_masks.txt
  - train/20_with_masks.txt
  - train/20_without_masks.txt
  - val_with_masks.txt
  - val_without_masks.txt
  - test_with_masks.txt
  - test_without_masks.txt
  - metadata.json (should contain a json that gets unique camera views and dates for each splitted txt file)
"""

"""
Finally, write a testing function to check that, for each split, one image-mask pair only exists in the original dataset once.
We have two sets that we need to check: "splits/split_by_timestamp/" and "splits/split_by_camera/".
For each set, we need to do two checks, one for with masks, one for without masks.
The first check is with masks using the following steps:
- merge the following files into one list and check that there is no duplicate entries in the merged list
  - train/100_with_masks.txt
  - val_with_masks.txt
  - test_with_masks.txt
- then, check that each entry in the merged list exists exactly once in "dataset/ijmond_seg/test/cropped/test_with_mask.txt"
The second check is without masks using the following steps:
- merge the following files into one list and check that there is no duplicate entries in the merged list
  - train/100_without_masks.txt
  - val_without_masks.txt
  - test_without_masks.txt
- then, check that each entry in the merged list exists exactly once in "dataset/ijmond_seg/test/cropped/test_without_mask.txt"
If any image-mask pair exists more than once in the original dataset, raise an error.
"""

"""
Behavior:
- Read listings:
  - dataset/ijmond_seg/test/cropped/test_with_mask.txt
  - dataset/ijmond_seg/test/cropped/test_without_mask.txt
- Extract camera (before "__") and timestamp (before "_frame" after "__").
- Print distributions (per-day and per-camera) for with/without masks.
- Create two split types: split_by_timestamp and split_by_camera.
- For training, produce 100/80/60/40/20 variants (earliest by timestamp).
- Write with/without mask files and a metadata.json per split folder.
- Run checks ensuring merged splits map back to originals exactly once.
"""

import json
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.getcwd())
LIST_DIR = os.path.join(ROOT, "dataset", "ijmond_seg", "test", "cropped")
WITH_MASK_FILE = os.path.join(LIST_DIR, "test_with_mask.txt")
WITHOUT_MASK_FILE = os.path.join(LIST_DIR, "test_without_mask.txt")
OUT_BASE = os.path.join(LIST_DIR, "splits")

CAMERAS_TEST = {"kooks_1", "hoogovens_6_7"}


def parse_listing(path: str, has_mask: Optional[bool] = None) -> List[Dict]:
    entries: List[Dict] = []
    if not os.path.exists(path):
        return entries
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            image = parts[0]
            mask = parts[1] if len(parts) > 1 else None
            if has_mask is not None:
                mask = mask if has_mask else None

            base = os.path.basename(image)
            camera: Optional[str] = None
            ts_dt: Optional[datetime] = None
            if "__" in base and "_frame" in base:
                camera, rest = base.split("__", 1)
                ts_str = rest.split("_frame", 1)[0]
                for fmt in ("%Y-%m-%dT%H-%M-%SZ", "%Y-%m-%dT%H-%M-%S"):
                    try:
                        ts_dt = datetime.strptime(ts_str, fmt)
                        break
                    except Exception:
                        continue

            entries.append({
                "image": image,
                "mask": mask,
                "camera": camera,
                "timestamp": ts_dt,
                "line": line,
            })
    return entries


def print_distribution(entries: List[Dict], label: str) -> None:
    by_day: Counter = Counter()
    by_camera: Counter = Counter()
    for e in entries:
        ts = e.get("timestamp")
        if ts:
            by_day[ts.date().isoformat()] += 1
        by_camera[e.get("camera") or "unknown"] += 1

    print(f"\nDistribution for {label} — total {len(entries)} entries")
    print("Images per day:")
    for day, cnt in sorted(by_day.items()):
        print(f"  {day}: {cnt}")
    print("Images per camera:")
    for cam, cnt in sorted(by_camera.items()):
        print(f"  {cam}: {cnt}")


def sort_by_timestamp(entries: List[Dict]) -> List[Dict]:
    return sorted(entries, key=lambda e: (e.get("timestamp") is None, e.get("timestamp")))


def split_timestamp(entries: List[Dict]) -> Dict[str, List[Dict]]:
    s = sort_by_timestamp(entries)
    n = len(s)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    train = s[:n_train]
    val = s[n_train : n_train + n_val]
    test = s[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def split_camera(entries: List[Dict]) -> Dict[str, List[Dict]]:
    k2 = sort_by_timestamp([e for e in entries if e.get("camera") == "kooks_2"])
    n = len(k2)
    n_train = int(n * 0.8)
    train = k2[:n_train]
    val = k2[n_train:]
    test = sort_by_timestamp([e for e in entries if e.get("camera") in CAMERAS_TEST])
    return {"train": train, "val": val, "test": test}


def write_list(path: str, entries: List[Dict], with_mask_file: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for e in entries:
            if with_mask_file:
                if e.get("mask"):
                    fh.write(f"{e['image']} {e['mask']}\n")
                else:
                    fh.write(f"{e['image']}\n")
            else:
                fh.write(f"{e['image']}\n")


def write_training_variants(train_entries: List[Dict], base_dir: str) -> None:
    train_dir = os.path.join(base_dir, "train")
    for pct in (100, 80, 60, 40, 20):
        take = int(len(train_entries) * pct / 100)
        # use the LAST `take` entries (most recent) rather than the first
        subset = train_entries[-take:] if take > 0 else []
        with_masks = [e for e in subset if e.get("mask")]
        without_masks = [e for e in subset if not e.get("mask")]
        write_list(os.path.join(train_dir, f"{pct}_with_masks.txt"), with_masks, True)
        write_list(os.path.join(train_dir, f"{pct}_without_masks.txt"), without_masks, False)


def write_val_test(ts: Dict[str, List[Dict]], base_dir: str) -> None:
    val_with = [e for e in ts["val"] if e.get("mask")]
    val_without = [e for e in ts["val"] if not e.get("mask")]
    write_list(os.path.join(base_dir, "val_with_masks.txt"), val_with, True)
    write_list(os.path.join(base_dir, "val_without_masks.txt"), val_without, False)

    test_with = [e for e in ts["test"] if e.get("mask")]
    test_without = [e for e in ts["test"] if not e.get("mask")]
    write_list(os.path.join(base_dir, "test_with_masks.txt"), test_with, True)
    write_list(os.path.join(base_dir, "test_without_masks.txt"), test_without, False)


def build_metadata(split_dir: str) -> None:
    metadata: Dict[str, Dict] = {}

    def extract_cam_date(line: str) -> Tuple[Optional[str], Optional[str]]:
        parts = line.split()
        if not parts:
            return None, None
        image = parts[0]
        base = os.path.basename(image)
        if "__" in base and "_frame" in base:
            cam, rest = base.split("__", 1)
            ts_str = rest.split("_frame", 1)[0]
            for fmt in ("%Y-%m-%dT%H-%M-%SZ", "%Y-%m-%dT%H-%M-%S"):
                try:
                    ts = datetime.strptime(ts_str, fmt)
                    return cam, ts.date().isoformat()
                except Exception:
                    continue
        return None, None

    for root, _, files in os.walk(split_dir):
        for fn in sorted(files):
            if not fn.endswith(".txt"):
                continue
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, split_dir)
            cams = set()
            dates = set()
            lines = load_lines(full)
            for l in lines:
                cam, date = extract_cam_date(l)
                if cam:
                    cams.add(cam)
                if date:
                    dates.add(date)
            metadata[rel] = {"cameras": sorted(cams), "dates": sorted(dates), "count": len(lines)}

    with open(os.path.join(split_dir, "metadata.json"), "w") as fh:
        json.dump(metadata, fh, indent=2)


def load_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r") as fh:
        return [l.strip() for l in fh if l.strip()]


def check_split_set(split_dir: str) -> None:
    # With masks
    files_with = [
        os.path.join(split_dir, "train", "100_with_masks.txt"),
        os.path.join(split_dir, "val_with_masks.txt"),
        os.path.join(split_dir, "test_with_masks.txt"),
    ]
    merged_with: List[str] = []
    for p in files_with:
        merged_with.extend(load_lines(p))
    if len(set(merged_with)) != len(merged_with):
        dup = [x for x, c in Counter(merged_with).items() if c > 1]
        raise ValueError(f"Duplicate entries in merged WITH_MASK split {split_dir}: {dup[:5]}")
    orig_with_counter = Counter(load_lines(WITH_MASK_FILE))
    missing = [m for m in merged_with if orig_with_counter.get(m, 0) == 0]
    multiple = [m for m in merged_with if orig_with_counter.get(m, 0) > 1]
    if missing:
        raise ValueError(f"WITH_MASK entries missing in original listing for {split_dir}: {missing[:5]}")
    if multiple:
        raise ValueError(f"WITH_MASK entries appear multiple times in original listing for {split_dir}: {multiple[:5]}")

    # Without masks
    files_wo = [
        os.path.join(split_dir, "train", "100_without_masks.txt"),
        os.path.join(split_dir, "val_without_masks.txt"),
        os.path.join(split_dir, "test_without_masks.txt"),
    ]
    merged_wo: List[str] = []
    for p in files_wo:
        merged_wo.extend(load_lines(p))
    if len(set(merged_wo)) != len(merged_wo):
        dup = [x for x, c in Counter(merged_wo).items() if c > 1]
        raise ValueError(f"Duplicate entries in merged WITHOUT_MASK split {split_dir}: {dup[:5]}")
    orig_wo_lines = load_lines(WITHOUT_MASK_FILE)
    orig_wo_images = [l.split()[0] for l in orig_wo_lines]
    orig_wo_counter = Counter(orig_wo_images)
    missing_wo = [m for m in merged_wo if orig_wo_counter.get(m.split()[0], 0) == 0]
    multiple_wo = [m for m in merged_wo if orig_wo_counter.get(m.split()[0], 0) > 1]
    if missing_wo:
        raise ValueError(f"WITHOUT_MASK entries missing in original listing for {split_dir}: {missing_wo[:5]}")
    if multiple_wo:
        raise ValueError(f"WITHOUT_MASK entries appear multiple times in original listing for {split_dir}: {multiple_wo[:5]}")


def make_splits(all_entries: List[Dict]) -> None:
    os.makedirs(OUT_BASE, exist_ok=True)
    ts = split_timestamp(all_entries)
    ts_dir = os.path.join(OUT_BASE, "split_by_timestamp")
    os.makedirs(ts_dir, exist_ok=True)
    write_training_variants(ts["train"], ts_dir)
    write_val_test(ts, ts_dir)
    build_metadata(ts_dir)

    cam = split_camera(all_entries)
    cam_dir = os.path.join(OUT_BASE, "split_by_camera")
    os.makedirs(cam_dir, exist_ok=True)
    write_training_variants(cam["train"], cam_dir)
    write_val_test(cam, cam_dir)
    build_metadata(cam_dir)


def main() -> None:
    with_mask_entries = parse_listing(WITH_MASK_FILE, has_mask=True)
    without_mask_entries = parse_listing(WITHOUT_MASK_FILE, has_mask=False)

    print_distribution(with_mask_entries, "with_masks")
    print_distribution(without_mask_entries, "without_masks")

    all_entries = with_mask_entries + without_mask_entries
    print(f"\nTotal merged entries: {len(all_entries)}")

    make_splits(all_entries)

    for split in ("split_by_timestamp", "split_by_camera"):
        split_dir = os.path.join(OUT_BASE, split)
        print(f"Running checks for {split_dir}...")
        check_split_set(split_dir)
    print("All checks passed.")


if __name__ == "__main__":
    main()
