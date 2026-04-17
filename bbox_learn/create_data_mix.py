"""
Creates mixed expert + citizen label txt files.

Expert splits are read from:
  dataset/ijmond_seg/test/cropped/splits/split_by_timestamp/train/

For each *_with_masks.txt file, the contents of
  dataset/ijmond_pseudo_masks/train_with_mask.txt
are appended and the result is written to:
  dataset/ijmond_expert_citizen_mix/<name>_mix_citizen.txt

For each *_without_masks.txt file, the contents of
  dataset/ijmond_pseudo_masks/train_without_mask.txt
are appended and the result is written to:
  dataset/ijmond_expert_citizen_mix/<name>_mix_citizen.txt
"""

import os

ROOT = os.path.abspath(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(ROOT, "dataset", "ijmond_seg", "test", "cropped", "splits", "split_by_timestamp", "train")
CITIZEN_WITH_MASK = os.path.join(ROOT, "dataset", "ijmond_pseudo_masks", "train_with_mask.txt")
CITIZEN_WITHOUT_MASK = os.path.join(ROOT, "dataset", "ijmond_pseudo_masks", "train_without_mask.txt")
OUT_DIR = os.path.join(ROOT, "dataset", "ijmond_expert_citizen_mix", "timestamp", "train")


def read_lines(path: str) -> list[str]:
    with open(path, "r") as fh:
        return [l.rstrip("\n") for l in fh if l.strip()]


def write_lines(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    citizen_with = read_lines(CITIZEN_WITH_MASK)
    citizen_without = read_lines(CITIZEN_WITHOUT_MASK)

    for fname in sorted(os.listdir(TRAIN_DIR)):
        if fname.endswith("_with_masks.txt"):
            expert_lines = read_lines(os.path.join(TRAIN_DIR, fname))
            mixed = expert_lines + citizen_with
            stem = fname[: -len(".txt")]
            out_path = os.path.join(OUT_DIR, f"{stem}_mix_citizen.txt")
            write_lines(out_path, mixed)
            print(f"Written {out_path}  ({len(expert_lines)} expert + {len(citizen_with)} citizen = {len(mixed)} total)")

        elif fname.endswith("_without_masks.txt"):
            expert_lines = read_lines(os.path.join(TRAIN_DIR, fname))
            mixed = expert_lines + citizen_without
            stem = fname[: -len(".txt")]
            out_path = os.path.join(OUT_DIR, f"{stem}_mix_citizen.txt")
            write_lines(out_path, mixed)
            print(f"Written {out_path}  ({len(expert_lines)} expert + {len(citizen_without)} citizen = {len(mixed)} total)")


if __name__ == "__main__":
    main()
