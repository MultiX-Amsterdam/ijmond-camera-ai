"""Pre-generate one corrupted version per image for deterministic robust evaluation.

For each image in ijmond_seg/test/cropped/images/, randomly selects one corruption
from {spatter, fog, brightness} at a random severity in [1, 3], applies it, and
saves the result to ijmond_seg/test/cropped/corrupt_images/.

Then writes new split txt files:
  experiment/expert_standard_val_corrupt.txt
  experiment/expert_standard_test_corrupt.txt
pointing to the corrupted images (masks unchanged).

Example usage:
  python create_corrupt_dataset.py
  python create_corrupt_dataset.py --data_root /path/to/bbox_learn/dataset --seed 42 --jobs 8
"""

import os
import sys
import random
import argparse
from functools import partial
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "imagecorruptions"))
from corrupt_images import corrupt_image, OutputType


CORRUPTIONS = ["spatter", "fog", "brightness"]
SEVERITIES = [1]

IMAGES_SUBDIR = "ijmond_seg/test/cropped/images"
CORRUPT_SUBDIR = "ijmond_seg/test/cropped/corrupt_images"

SPLIT_FILES = {
    "val": "experiment/expert_standard_val.txt",
    "test": "experiment/expert_standard_test.txt",
}

CORRUPT_SPLIT_FILES = {
    "val": "experiment/expert_standard_val_corrupt.txt",
    "test": "experiment/expert_standard_test_corrupt.txt",
}


def _corrupt_one(task, data_root, images_dir, corrupt_dir):
    """Apply one pre-chosen corruption to a single image.

    Parameters
    ----------
    task : tuple of (str, str, int)
        (img_rel, corruption_name, severity) — the relative image path, the
        chosen corruption name, and the chosen severity level.
    data_root : str
        Absolute path to the dataset root directory.
    images_dir : str
        Absolute path to the source images directory.
    corrupt_dir : str
        Absolute path to the output corrupted images directory.
    """
    img_rel, corruption, severity = task
    stem, ext = os.path.splitext(os.path.basename(img_rel))
    corrupt_filename = f"{stem}_{corruption}_{severity}{ext}"
    if os.path.exists(os.path.join(corrupt_dir, corrupt_filename)):
        return
    img_full = os.path.join(data_root, img_rel)
    corrupt_image(
        img_full,
        images_dir,
        corrupt_dir,
        OutputType.FILENAME,
        [corruption],
        [severity],
    )


def get_corrupt_rel(img_rel, corruption, severity):
    """Compute the relative path of a corrupted image under CORRUPT_SUBDIR.

    Parameters
    ----------
    img_rel : str
        Relative image path, e.g. 'ijmond_seg/test/cropped/images/foo.jpg'.
    corruption : str
        Name of the corruption applied.
    severity : int
        Severity level of the corruption.

    Returns
    -------
    str
        Relative path of the corrupted image, e.g.
        'ijmond_seg/test/cropped/corrupt_images/foo_spatter_2.jpg'.
    """
    stem, ext = os.path.splitext(os.path.basename(img_rel))
    corrupt_filename = f"{stem}_{corruption}_{severity}{ext}"
    return os.path.join(CORRUPT_SUBDIR, corrupt_filename).replace("\\", "/")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pre-generate corrupted val/test images for deterministic robust evaluation. "
            "Each image gets exactly one corruption randomly sampled from "
            "{spatter, fog, brightness} at a random severity in [1, 3]."
        )
    )
    parser.add_argument(
        "--data_root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"),
        help="Path to bbox_learn/dataset/ (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: %(default)s)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = os.path.join(args.data_root, IMAGES_SUBDIR)
    corrupt_dir = os.path.join(args.data_root, CORRUPT_SUBDIR)
    os.makedirs(corrupt_dir, exist_ok=True)

    # Read all split txt files and collect unique image relative paths.
    split_lines = {}
    unique_image_rels = set()
    for split, rel_path in SPLIT_FILES.items():
        txt_path = os.path.join(args.data_root, rel_path)
        with open(txt_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        split_lines[split] = [line.split(" ") for line in lines]
        for parts in split_lines[split]:
            unique_image_rels.add(parts[0])

    # Assign one (corruption, severity) pair per unique image.
    # Sort for determinism under a fixed seed.
    corruption_map = {}
    for img_rel in sorted(unique_image_rels):
        corruption_map[img_rel] = (
            random.choice(CORRUPTIONS),
            random.choice(SEVERITIES),
        )

    print(f"Corrupting {len(corruption_map)} unique images -> {corrupt_dir}")

    tasks = [
        (img_rel, corruption, severity)
        for img_rel, (corruption, severity) in corruption_map.items()
    ]

    worker = partial(
        _corrupt_one,
        data_root=args.data_root,
        images_dir=images_dir,
        corrupt_dir=corrupt_dir,
    )

    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            pool.map(worker, tasks)
    else:
        for task in tasks:
            worker(task)

    print("Done corrupting images. Writing corrupt split txt files ...")

    # Write new txt files with corrupted image paths; mask paths are unchanged.
    for split, lines_parts in split_lines.items():
        output_txt = os.path.join(args.data_root, CORRUPT_SPLIT_FILES[split])
        with open(output_txt, "w") as f:
            for parts in lines_parts:
                img_rel = parts[0]
                corruption, severity = corruption_map[img_rel]
                corrupt_img_rel = get_corrupt_rel(img_rel, corruption, severity)
                new_parts = [corrupt_img_rel] + parts[1:]
                f.write(" ".join(new_parts) + "\n")
        print(f"  Written: {output_txt}")


if __name__ == "__main__":
    main()
