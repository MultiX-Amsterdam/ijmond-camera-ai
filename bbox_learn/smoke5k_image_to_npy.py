"""
Convert images in the dataset to npy files.
"""

from util.util import convert_all_images_to_npy

convert_all_images_to_npy("dataset/smoke5k/test/img", "dataset/smoke5k/test/img_npy")
convert_all_images_to_npy("dataset/smoke5k/test/gt", "dataset/smoke5k/test/gt_npy")
convert_all_images_to_npy("dataset/smoke5k/train/img", "dataset/smoke5k/train/img_npy")
convert_all_images_to_npy("dataset/smoke5k/train/gt", "dataset/smoke5k/train/gt_npy")
