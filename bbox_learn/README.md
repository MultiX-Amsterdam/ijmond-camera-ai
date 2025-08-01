# Weakly Semi-Supervised Learning

This folder contains a weakly semi-supervised learning pipeline that uses bounding boxes and unlabeled images (highly likely to contain smoke emissions) to train a segmentation model, which outputs pixel-level masks for industrial smoke segmentation. The segmentation model is first pretrained on a dataset from another context (a combination of wildfire detection and synthetic smoke) and then fine-tuned in our context using bounding boxes and unlabeled smoke images.

Prepare IJmond bounding boxes for training:
```sh
python download_ijmond_bbox_images.py dataset/ijmond_bbox/bbox_labels_1_aug_2025.json
python filter_aggr_bbox.py dataset/ijmond_bbox/bbox_labels_1_aug_2025.json dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json
```

Test if the IJmond bounding boxes can be loaded:
```sh
python ijmond_bbox_dataset.py dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img_npy/
```

Prepare SMOKE5K for training:
```sh
python create_smoke5k_metadata_and_npy.py dataset/smoke5k/
```

Create pseudo masks using the IJmond bounding boxes and save the masks in the `dataset/ijmond_pseudo_masks/` path:
```sh
python create_pseudo_masks.py dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img_npy/
```
