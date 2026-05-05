"""
This file shows dataset statistics.
"""

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone


def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

ROOT = os.path.abspath(os.getcwd())
BBOX_DIR = os.path.join(ROOT, "dataset", "ijmond_bbox")
VID_DIR = os.path.join(ROOT, "dataset", "ijmond_vid")
PSEUDO_DIR = os.path.join(ROOT, "dataset", "ijmond_pseudo_masks")
SMOKE5K_DIR = os.path.join(ROOT, "dataset", "smoke5k")


def count_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# 1. Citizen bounding box contributions
# ---------------------------------------------------------------------------
section("1. Citizen bounding box contributions")
with open(os.path.join(BBOX_DIR, "bbox_labels_1_aug_2025.json")) as f:
    bbox_raw = json.load(f)
citizen_bboxes = bbox_raw["data"] if isinstance(bbox_raw, dict) else bbox_raw
print(f"Total citizen bbox contributions: {len(citizen_bboxes)}")
frame_ts = [e["frame_timestamp"] for e in citizen_bboxes if "frame_timestamp" in e]
label_ts = [e["label_update_time"] for e in citizen_bboxes if "label_update_time" in e]
if frame_ts:
    print(f"  Frame capture time range:  {fmt_ts(min(frame_ts))} → {fmt_ts(max(frame_ts))}")
if label_ts:
    print(f"  Label submission time range: {fmt_ts(min(label_ts))} → {fmt_ts(max(label_ts))}")

# ---------------------------------------------------------------------------
# 2. Aggregated bounding boxes
# ---------------------------------------------------------------------------
section("2. Aggregated bounding boxes (after citizen annotation aggregation)")
with open(os.path.join(BBOX_DIR, "filtered_bbox_labels_1_aug_2025.json")) as f:
    filtered = json.load(f)
total_frames = len(filtered)
frames_with_bbox = sum(1 for e in filtered if e.get("bbox"))
frames_without_bbox = total_frames - frames_with_bbox
total_individual_bboxes = sum(len(e["bbox"]) for e in filtered if e.get("bbox"))
print(f"Total aggregated frames:                   {total_frames}")
print(f"  Frames with smoke bbox:                  {frames_with_bbox}")
print(f"  Frames without smoke (no bbox / null):   {frames_without_bbox}")
print(f"Total individual bboxes across all frames: {total_individual_bboxes}")

# ---------------------------------------------------------------------------
# 3. Unlabeled training videos
# ---------------------------------------------------------------------------
section("3. Unlabeled training videos")
with open(os.path.join(VID_DIR, "metadata_ijmond_jan_22_2024.json")) as f:
    vid_meta = json.load(f)
n_videos = len(vid_meta)
print(f"Number of unlabeled training videos: {n_videos}")
start_times = [v["start_time"] for v in vid_meta if "start_time" in v]
if start_times:
    print(f"  Video date range: {fmt_ts(min(start_times))} → {fmt_ts(max(start_times))}")

FRAMES_PER_VIDEO = 36
SECONDS_PER_FRAME = 5
frames_total = n_videos * FRAMES_PER_VIDEO
duration_minutes = n_videos * FRAMES_PER_VIDEO * SECONDS_PER_FRAME / 60
print(f"  Each video: {FRAMES_PER_VIDEO} frames @ {SECONDS_PER_FRAME}s intervals = "
      f"{FRAMES_PER_VIDEO * SECONDS_PER_FRAME}s ({FRAMES_PER_VIDEO * SECONDS_PER_FRAME / 60:.1f} min)")
print(f"  Total frames (estimated): {frames_total}")
print(f"  Total footage (estimated): {duration_minutes:.1f} min")

# Actual frame count from unlabeled.txt
n_frames_actual = count_lines(os.path.join(VID_DIR, "unlabeled.txt"))
print(f"  Actual frame count in unlabeled.txt: {n_frames_actual}")

# ---------------------------------------------------------------------------
# 4. Camera distribution in unlabeled training videos
# ---------------------------------------------------------------------------
section("4. Camera distribution in unlabeled training videos")
cam_counter: Counter = Counter(v["camera_id"] for v in vid_meta)
for cam_id, cnt in sorted(cam_counter.items()):
    print(f"  camera_id {cam_id}: {cnt} videos")

# ---------------------------------------------------------------------------
# 5. Views per camera in unlabeled training videos
# ---------------------------------------------------------------------------
section("5. Views per camera in unlabeled training videos")
cam_views: dict = defaultdict(set)
cam_view_counts: dict = defaultdict(Counter)
for v in vid_meta:
    cam_views[v["camera_id"]].add(v["view_id"])
    cam_view_counts[v["camera_id"]][v["view_id"]] += 1
for cam_id in sorted(cam_views):
    views = sorted(cam_views[cam_id])
    print(f"  camera_id {cam_id}: {len(views)} unique view(s): {views}")
    for view_id in views:
        print(f"    view_id {view_id}: {cam_view_counts[cam_id][view_id]} videos")

# ---------------------------------------------------------------------------
# 6. Start time distribution of unlabeled training videos
# ---------------------------------------------------------------------------
section("6. Start time distribution of unlabeled training videos")
by_year_month: Counter = Counter()
for v in vid_meta:
    dt = datetime.fromtimestamp(v["start_time"], tz=timezone.utc)
    by_year_month[dt.strftime("%Y-%m")] += 1
print("  Videos per month (UTC):")
for ym, cnt in sorted(by_year_month.items()):
    print(f"    {ym}: {cnt}")

# ---------------------------------------------------------------------------
# 7. Pseudo-labeled masks (SAM)
# ---------------------------------------------------------------------------
section("7. Pseudo-labeled masks generated by SAM")
n_pseudo_with = count_lines(os.path.join(PSEUDO_DIR, "train_with_mask.txt"))
n_pseudo_without = count_lines(os.path.join(PSEUDO_DIR, "train_without_mask.txt"))
print(f"  train_with_mask.txt (smoke frames):     {n_pseudo_with}")
print(f"  train_without_mask.txt (non-smoke):     {n_pseudo_without}")
print(f"  Total:                                  {n_pseudo_with + n_pseudo_without}")

# ---------------------------------------------------------------------------
# 8. Smoke5K dataset
# ---------------------------------------------------------------------------
section("8. Smoke5K dataset")
n_train_smoke5k = count_lines(os.path.join(SMOKE5K_DIR, "train", "train.txt"))
n_test_smoke5k = count_lines(os.path.join(SMOKE5K_DIR, "test", "test.txt"))
print(f"  train.txt: {n_train_smoke5k}")
print(f"  test.txt:  {n_test_smoke5k}")
print(f"  Total:     {n_train_smoke5k + n_test_smoke5k}")