# Weakly Semi-Supervised Learning

This folder contains a weakly semi-supervised learning pipeline that uses bounding boxes and unlabeled images (highly likely to contain smoke emissions) to train a segmentation model, which outputs pixel-level masks for industrial smoke segmentation. The segmentation model is first pretrained on a dataset from another context (a combination of wildfire detection and synthetic smoke) and then fine-tuned in our context using bounding boxes and unlabeled smoke images.

Prepare IJmond bounding boxes for training. This will download images to the `dataset/ijmond_bbox/img` folder, create a `filtered_bbox_labels_1_aug_2025.json` file with filtered and aggregated bounding boxes, and create `.npy` files in the `dataset/ijmond_bbox/img_npy` folder, and create debugging images to the `dataset/ijmond_bbox/debug` folder.
```sh
python download_ijmond_bbox_images.py dataset/ijmond_bbox/bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img
python filter_aggr_bbox_and_create_npy.py dataset/ijmond_bbox/bbox_labels_1_aug_2025.json dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json dataset/ijmond_bbox/
```

Test if the IJmond bounding boxes can be loaded. This will create a `debug_plot_ijmondbox.png` file for debugging.
```sh
python ijmond_bbox_dataset.py dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img_npy/
```

Prepare SMOKE5K for training. This will create `.npy` files and also metadata txt files in `dataset/smoke5k/`.
```sh
python create_smoke5k_metadata_and_npy.py dataset/smoke5k/
```

Check if the SMOKE5K dataset can be loaded. This will create `debug_plot_smoke5k_test_img.png` and `debug_plot_smoke5k_test_gt.png` files for debugging.
```sh
python smoke_dataset.py dataset/smoke5k/test/test.txt dataset/smoke5k/test/ smoke5k_test
```

Create pseudo masks and metadata txt files (one with masks, one without masks) using the IJmond bounding boxes and save the masks in the `dataset/ijmond_pseudo_masks/` path:
```sh
python create_pseudo_masks.py dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img_npy/
```

Check if the IJmond pseudo masks dataset can be loaded. This will create `debug_plot_ijmond_pseudo_masks_with_mask_img.png` and `debug_plot_ijmond_pseudo_masks_with_mask_gt.png` files for debugging.
```sh
python smoke_dataset.py dataset/ijmond_pseudo_masks/train_with_mask.txt dataset/ijmond_pseudo_masks/ ijmond_pseudo_mask_with_mask
```

Prepare the unlabeled data from IJmond Videos. This will download videos, extract frames, and create a metadata txt file.
```sh
cd dataset/ijmond_vid/
python download_videos.py
python extract_frames.py
```

Check if the unlabeled IJmond video dataset can be loaded. This will create the `debug_plot_ijmond_vid_unlabeled_img.png` file for debugging.
```sh
python smoke_dataset.py dataset/ijmond_vid/unlabeled.txt dataset/ijmond_vid/ ijmond_vid_unlabeled
```
