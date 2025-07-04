# Weakly Semi-Supervised Learning

This folder contains a weakly semi-supervised learning pipeline that uses bounding boxes and unlabeled images (highly likely to contain smoke emissions) to train a segmentation model, which outputs pixel-level masks for industrial smoke segmentation. The segmentation model is first pretrained on a dataset from another context (a combination of wildfire detection and synthetic smoke) and then fine-tuned in our context using bounding boxes and unlabeled smoke images.

Prepare IJmond bounding boxes for training:
```sh
cd bbox_learn/dataset/ijmond_bbox
python download_bbox.py bbox_labels_4_july_2025.json
cd ../../
python filter_aggr_bbox.py dataset/ijmond_bbox/bbox_labels_4_july_2025.json dataset/ijmond_bbox/filtered_bbox_labels_4_july_2025.json
```

Test if the IJmond bounding boxes can be loaded:
```sh
python ijmond_bbox_dataset.py dataset/ijmond_bbox/filtered_bbox_labels_4_july_2025.json dataset/ijmond_bbox/img_npy/
```

Prepare SMOKE5K for training:
```sh
python smoke5k_image_to_npy.py
```