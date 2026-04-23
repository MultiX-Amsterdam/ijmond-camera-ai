"""
Creates all experiment dataset txt files under dataset/experiment/ using the
dataset abbreviations defined in the README "Datasets" section.

All 27 files are written flat into dataset/experiment/ with names matching
the dataset abbreviations. Each file contains paths relative to the dataset
root (bbox_learn/dataset/) so they work directly with SemiSmokeDataset:

    SemiSmokeDataset(root=data_root, id_path="experiment/<name>.txt")

Files produced
--------------
smoke5k         (2)  smoke5k_train.txt, smoke5k_test.txt
citizen         (2)  citizen_with_mask.txt, citizen_without_mask.txt
unlabeled       (1)  unlabeled.txt
expert val/test (2)  expert_standard_val.txt,
                     expert_standard_test.txt
expert train   (10)  expert_standard_train_{P}_{with,without}_masks.txt
                     for P in 100, 80, 60, 40, 20
mix train      (10)  mix_standard_train_{P}_{with,without}_masks.txt
                     for P in 100, 80, 60, 40, 20

Total: 27 files.

Usage:
    python create_experiment_datasets.py
"""

import os
import shutil

ROOT = os.path.abspath(os.path.dirname(__file__))
DATASET = os.path.join(ROOT, "dataset")
EXP_DIR = os.path.join(DATASET, "experiment")

SEG_CROPPED = "ijmond_seg/test/cropped/"
SEG_CAM_BAL = SEG_CROPPED + "splits/split_standard/"
PERCENTAGES = [100, 80, 60, 40, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_lines(path: str) -> list[str]:
    with open(path, "r") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def write_lines(name: str, lines: list[str]) -> None:
    """Write *lines* to experiment/<name>.txt and print a summary."""
    path = os.path.join(EXP_DIR, name + ".txt")
    os.makedirs(EXP_DIR, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  {name}.txt  ({len(lines)} lines)")


def prefix_paths(lines: list[str], prefix: str) -> list[str]:
    """Prepend *prefix* to every non-empty, non-None token on each line."""
    result = []
    for line in lines:
        parts = line.split(" ")
        fixed = [prefix + p if p and p != "None" else p for p in parts]
        result.append(" ".join(fixed))
    return result


def load(src_rel: str, prefix: str) -> list[str]:
    """Read source txt (relative to DATASET) and fix path prefixes."""
    return prefix_paths(
        read_lines(os.path.join(DATASET, src_rel)),
        prefix,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Remove old experiment directory so stale files do not accumulate
    if os.path.exists(EXP_DIR):
        shutil.rmtree(EXP_DIR)

    print("=== smoke5k ===")
    write_lines("smoke5k_train", load("smoke5k/train/train.txt", "smoke5k/train/"))
    write_lines("smoke5k_test",   load("smoke5k/test/test.txt",   "smoke5k/test/"))

    print("\n=== citizen ===")
    citizen_with    = load("ijmond_pseudo_masks/train_with_mask.txt",    "ijmond_pseudo_masks/")
    citizen_without = load("ijmond_pseudo_masks/train_without_mask.txt", "ijmond_pseudo_masks/")
    write_lines("citizen_with_mask",    citizen_with)
    write_lines("citizen_without_mask", citizen_without)

    print("\n=== unlabeled ===")
    write_lines("unlabeled", load("ijmond_vid/unlabeled.txt", "ijmond_vid/"))

    print("\n=== expert_standard_val ==="
          " (combines val_with_masks + val_without_masks)")
    bal_val_lines = (
        load(SEG_CAM_BAL + "val_with_masks.txt",    SEG_CROPPED)
        + load(SEG_CAM_BAL + "val_without_masks.txt", SEG_CROPPED)
    )
    write_lines("expert_standard_val", bal_val_lines)

    print("\n=== expert_standard_test ==="
          " (combines test_with_masks + test_without_masks)")
    bal_test_lines = (
        load(SEG_CAM_BAL + "test_with_masks.txt",    SEG_CROPPED)
        + load(SEG_CAM_BAL + "test_without_masks.txt", SEG_CROPPED)
    )
    write_lines("expert_standard_test", bal_test_lines)

    print("\n=== expert_standard_train_{P}_{with,without}_masks ===")
    expert_standard_with: dict[str, list[str]] = {}
    expert_standard_without: dict[str, list[str]] = {}
    for p in PERCENTAGES:
        expert_standard_with[str(p)] = load(f"{SEG_CAM_BAL}train/{p}_with_masks.txt", SEG_CROPPED)
        expert_standard_without[str(p)] = load(f"{SEG_CAM_BAL}train/{p}_without_masks.txt", SEG_CROPPED)
        write_lines(f"expert_standard_train_{p}_with_masks",    expert_standard_with[str(p)])
        write_lines(f"expert_standard_train_{p}_without_masks", expert_standard_without[str(p)])

    print("\n=== mix_standard_train_{P}_{with,without}_masks (expert_standard + citizen) ===")
    for p in PERCENTAGES:
        write_lines(
            f"mix_standard_train_{p}_with_masks",
            expert_standard_with[str(p)] + citizen_with,
        )
        write_lines(
            f"mix_standard_train_{p}_without_masks",
            expert_standard_without[str(p)] + citizen_without,
        )

    total = len(os.listdir(EXP_DIR))
    print(f"\nDone. {total} files written to bbox_learn/dataset/experiment/")


if __name__ == "__main__":
    main()
