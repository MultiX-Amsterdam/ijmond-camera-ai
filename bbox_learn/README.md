# Weakly Semi-Supervised Learning

This folder contains a weakly semi-supervised learning pipeline that uses bounding boxes and unlabeled images (highly likely to contain smoke emissions) to train a segmentation model, which outputs pixel-level masks for industrial smoke segmentation. The segmentation model is first pretrained on a dataset from another context (a combination of wildfire detection and synthetic smoke) and then fine-tuned in our context using bounding boxes and unlabeled smoke images.

## Data Preparation

### Prepare citizen-labeled IJmond bounding boxes

Prepare IJmond bounding boxes for training. This will download images to the `dataset/ijmond_bbox/img` folder, create a `filtered_bbox_labels_1_aug_2025.json` file with filtered and aggregated bounding boxes, and create debugging images to the `dataset/ijmond_bbox/debug` folder.
```sh
python download_ijmond_bbox_images.py dataset/ijmond_bbox/bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img
python filter_aggr_bbox.py dataset/ijmond_bbox/bbox_labels_1_aug_2025.json dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json dataset/ijmond_bbox/
```

Test if the IJmond bounding boxes can be loaded. This will create a `debug_plot_ijmondbox.png` file for debugging.
```sh
python ijmond_bbox_dataset.py dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img/
```

### Prepare SMOKE5K data for pretraining

Prepare SMOKE5K for training. This will create metadata txt files in `dataset/smoke5k/`.
```sh
python create_smoke5k_metadata.py dataset/smoke5k/
```

Check if the SMOKE5K dataset can be loaded. This will create `debug_plot_smoke5k_test.png`, `debug_plot_smoke5k_test_transformed.png`, `debug_plot_smoke5k_train.png`, and `debug_plot_smoke5k_train_transformed.png` files for debugging.
```sh
python smoke_dataset.py dataset/smoke5k/test/test.txt dataset/smoke5k/test/ smoke5k_test
python smoke_dataset.py dataset/smoke5k/train/train.txt dataset/smoke5k/train/ smoke5k_train
```

### Create IJmond pseudo masks based on the bounding boxes

We use Segment Anything (SAM) to create the pseudo masks. Before doing this, run the following on the terminal to install SAM. You need to be in the `bbox_learn` directory.
```sh
pip install git+https://github.com/facebookresearch/segment-anything.git
wget -P https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

Create pseudo masks and metadata txt files (one with masks, one without masks) using the IJmond bounding boxes and save the masks in the `dataset/ijmond_pseudo_masks/` path. This will create `debug_plot_pseudo_masks.png` file for debugging.
```sh
python create_pseudo_masks.py dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img/
```

Check if the IJmond pseudo masks dataset can be loaded. This will create `debug_plot_ijmond_pseudo_masks_with_mask.png` and `debug_plot_ijmond_pseudo_masks_with_mask_transformed.png` files for debugging.
```sh
python smoke_dataset.py dataset/ijmond_pseudo_masks/train_with_mask.txt dataset/ijmond_pseudo_masks/ ijmond_pseudo_mask_with_mask
```

### Prepare unlabeled data

Prepare the unlabeled data from IJmond Videos. This will download videos, extract frames, and create a metadata txt file.
```sh
cd dataset/ijmond_vid/
python download_videos.py
python extract_frames.py
```

Check if the unlabeled IJmond video dataset can be loaded. This will create the `debug_plot_ijmond_vid_unlabeled_img.png` and `debug_plot_ijmond_vid_unlabeled_img_transformed.png` files for debugging.
```sh
python smoke_dataset.py dataset/ijmond_vid/unlabeled.txt dataset/ijmond_vid/ ijmond_vid_unlabeled
```

### Prepare expert-labeled IJmond segmentation masks and splits

Prepare the IJmond segmentation dataset. You need to first get the dataset with the following structure, which requires moving images from the downloaded Roboflow data into an `images` folder, as shown below:
```sh
└── dataset # the root folder
    └── ijmond_seg # the folder that contains the IJmond segmentation dataset
        ├── test
            └── images # all camera images
                ├── XXX.jpg
                └── ...
        └── _annotations.coco.json # the annotation file
```

Then, run a script to crop the images. This will first create segmentation masks (under `dataset/ijmond_seg/test/masks/`) and then crop the large panoramas into smaller ones (under `dataset/ijmond_seg/test/cropped/`).
```sh
python create_ijmond_seg_masks.py dataset/ijmond_seg/test/images/ dataset/ijmond_seg/test/_annotations.coco.json dataset/ijmond_seg/test/masks/
python crop_ijmond_seg.py dataset/ijmond_seg/test/images/ dataset/ijmond_seg/test/_annotations.coco.json dataset/ijmond_seg/test/masks/ dataset/ijmond_seg/test/cropped/
```

So, after that, the file structure should look like below:
```sh
└── dataset # the root folder
    └── ijmond_seg # the folder that contains the IJmond segmentation dataset
        └── test
            ├── cropped # all cropped images and masks
                ├── images # all cropped camera images
                    ├── XXX.jpg
                    └── ...
                ├── masks # all cropped masks
                    ├── XXX.png
                    └── ...
                ├── test_with_mask.txt # paths for image-mask pairs (with masks)
                ├── test_without_mask.txt # paths for image-mask pairs (with no masks)
                └── metadata.json # metadata for each cropped image
            ├── images # all camera images
                ├── XXX.jpg
                └── ...
            ├── masks # all masks
                ├── XXX.png
                └── ...
            └── _annotations.coco.json # the annotation file
```

Finally, split the IJmond dataset into training, validation, and test sets. Check the documentation in the `split_ijmond_seg.py` file to understand how we split the data.
```sh
python split_ijmond_seg.py
```

After that, there will be a new `splits` folder under the `dataset/ijmond_seg/test/cropped/` directory. The `splits` folder has two subfolders that indicate two different types of splits: `split_by_camera` and `split_by_timestamp`. We have three cameras: "kooks_1", "kooks_2", and "hoogovens_6_7". For the training set, we further split then into 100/80/60/40/20% to simulate different amount of available training data. These percentages are the "last" part in the training set according to sorted timestamps to ensure that the timestamps, when considered together with the validation and test sets, are continuous. The also further seperate them into with and without masks to specifically get negative samples for training. Below is the explaination for split by camera:
```sh
└── split_by_camera # the split based on camera views
    ├── train # training set, which is the first 80% of "kooks_2" sorted by timestamps
        ├── 100_with_masks.txt # 100% of the training set with masks
        ├── 100_without_masks.txt
        ├── 80_with_masks.txt # last 80% of the training set with masks
        ├── 80_without_masks.txt
        ├── 60_with_masks.txt # last 60% of the training set with masks
        ├── 60_without_masks.txt
        ├── 40_with_masks.txt # last 40% of the training set with masks
        ├── 40_without_masks.txt
        ├── 20_with_masks.txt # last 20% of the training set with masks
        └── 20_without_masks.txt
    ├── val_with_masks.txt # validation set with masks, which is the rest of 20% of "kooks_2"
    ├── val_without_masks.txt
    ├── test_with_masks.txt # test set with masks, which is 100% of "hoogovens_6_7" and "kooks_1"
    ├── test_without_masks.txt
    └── metadata.json # the coverage of camera views and dates for each txt file
```

Below is the explaination for split by timestamp:
```sh
└── split_by_timestamp # the split based on timestamps
    ├── train # training set, which is the first 70% sorted by timestamps
        ├── 100_with_masks.txt # 100% of the training set with masks
        ├── 100_without_masks.txt
        ├── 80_with_masks.txt # last 80% of the training set with masks
        ├── 80_without_masks.txt
        ├── 60_with_masks.txt # last 60% of the training set with masks
        ├── 60_without_masks.txt
        ├── 40_with_masks.txt # last 40% of the training set with masks
        ├── 40_without_masks.txt
        ├── 20_with_masks.txt # last 20% of the training set with masks
        └── 20_without_masks.txt
    ├── val_with_masks.txt # validation set with masks, which is the next 10% after training set
    ├── val_without_masks.txt
    ├── test_with_masks.txt # test set with masks, which is the next 20% after validation set
    ├── test_without_masks.txt
    └── metadata.json # the coverage of camera views and dates for each txt file
```

You can check if the cropped IJmond segmentation dataset can be loaded. This will create the `debug_plot_ijmond_seg_cropped_train_with_mask_20.png` and `debug_plot_ijmond_seg_cropped_train_with_mask_20_transformed.png` files for debugging.
```sh
python smoke_dataset.py dataset/ijmond_seg/test/cropped/splits/split_by_timestamp/train/20_with_masks.txt dataset/ijmond_seg/test/cropped/ ijmond_seg_cropped_train_with_mask_20
```

And lastly, run the following to mix expert and citizen data for experiments:
```sh
python create_data_mix.py
```

## Experiment Settings

For experiments, all models should first load the large-scale pretrained weights (e.g., DINOv2), which depends on the model implementation. In this experiment, we use UniMatch-V2. Then, all models should first be pretrained again using the `smoke5k` dataset to simulate the situation that we have some prior model in a similar problem domain (smoke segmentaion) to begin with. We call this the `Smoke5K-pretrained-UniMatch-V2` model.

Then, depending on the research question, we finetune the model (or not) based on specific sets and evaluate the model.

### The 10% negative samples rule

During the finetuning stage, we always use the full set of images with masks (i.e., positive samples) and then combine it with some randomly selected negative samples (10% of the batch size) from the set without masks. For example, if we are using 100% of the training data, and the batch size is 40 when looping the dataloader of the `100_with_masks.txt` file, we will randomly pick 4 negative samples (10% of the batch size) from `100_without_masks.txt` and add these negative samples to the batch when performing one batch gradient descent step. Same thing applies for the `ijmond_pseudo_masks` dataset, which has `train_with_mask.txt` and `train_without_mask.txt`. The reason of doing this (not using too many negative samples) is because we do not want the model to just predict `no smoke` for all the pixels to get a low loss during training.

### The unlabeled data sampling rule

When using unlabeled data, we randomly sample a set of unlabeled images during training for each iteraton (i.e., each batch gradient descent step) to reduce the computation time. The number of unlabeled images is the same as labeled images, which is the same implementation as in the [UniMatchV2 paper](https://arxiv.org/abs/2410.10777).

### Datasets

For simplicity, we use the following dataset abbreviations with their paths. For the citizen-contributed data, we have:
- `citizen_with_mask`:
  - `ijmond_pseudo_masks/train_with_mask.txt`
- `citizen_without_mask`:
  - `ijmond_pseudo_masks/train_without_mask.txt`

For unlabeled data, we have:
- `unlabeled`:
  - `ijmond_vid/unlabeled.txt`

For expert-labeled data, we have the followings for validation and testing:
- `expert_timestamp_val_with_masks`:
  - `ijmond_seg/test/cropped/splits/split_by_timestamp/val_with_masks.txt`
- `expert_timestamp_val_without_masks`
  - `ijmond_seg/test/cropped/splits/split_by_timestamp/val_without_masks.txt`
- `expert_timestamp_test_with_masks`:
  - `ijmond_seg/test/cropped/splits/split_by_timestamp/test_with_masks.txt`
- `expert_timestamp_test_without_masks`
  - `ijmond_seg/test/cropped/splits/split_by_timestamp/test_without_masks.txt`

For expert-labeled data, we have the following timestamp and camera splits for training, where placeholder `{P}` can be `100`, `80`, `60`, `40`, or `20`, representing the amount of available training data.
- `expert_timestamp_train_{P}_with_masks`:
  - `ijmond_seg/test/cropped/splits/split_by_timestamp/train/{P}_with_masks.txt`
- `expert_timestamp_train_{P}_without_masks`:
  - `ijmond_seg/test/cropped/splits/split_by_timestamp/train/{P}_without_masks.txt`

For example, dataset `expert_timestamp_train_100_with_masks` has path `ijmond_seg/test/cropped/splits/split_by_timestamp/train/100_with_masks.txt`.

### RQ1: How useful is citizen-contributed weak labels?

IMPORTANT: All models start with the `Smoke5K-pretrained-UniMatch-V2` model, which loads large-scale `UniMatch-V2` pretrained weights and then pretrained again on the `smoke5k` dataset.

For this research question, we have the following base models:
- `M-zeroshot`: no finetuning, which is exactly the `Smoke5K-pretrained-UniMatch-V2` model
- `M-citizen`: finetuned using weakly-labeled and unlabeled datasets below:
  - `citizen_with_mask`
  - `citizen_without_mask` (using the 10% negative samples rule as mentioned before)
  - `unlabeled`

We only use the timestamp split for this research question with one additional model:
- `M-expert`: finetuned using expert-labeled and unlabeled datasets below:
  - `expert_timestamp_train_100_with_masks`
  - `expert_timestamp_train_100_without_masks` (using the 10% negative samples rule as mentioned before)
  - `unlabeled`

The expert model `M-expert` serves as a reference point in the situation without the help of citizens. The citizen model `M-citizen` represents the situation with only the help from citizens. Model `M-zeroshot` is the zero-shot case without any finetuning. We also have the expert-citizen collaboration model, which is `M-mix-100` in the next research question. By comparing the performance of these models, we know if citizen-contributed data is useful and to what extent.

In this setting, `M-zeroshot` will be the lower bound of performance, and `M-mix-100` will be the upper bound of performance.

### RQ2: How much contribution from the experts is needed?

IMPORTANT: All models start with the `Smoke5K-pretrained-UniMatch-V2` model, which loads large-scale `UniMatch-V2` pretrained weights and then pretrained again on the `smoke5k` dataset.

For this research question, we also only use the timestamp split. We need to first mix the expert and citizen data. We list the mix below and give them new names:
- `mix_timestamp_train_{P}_with_masks`: combines the followiing
  - `expert_timestamp_train_{P}_with_masks`
  - `citizen_with_mask`
- `mix_timestamp_train_{P}_without_masks`: combines the followiing
  - `expert_timestamp_train_{P}_without_masks`
  - `citizen_without_mask`

We use the following models for this experiment:
- `M-mix-{P}`:
  - finetuned using the following datasets:
    - `mix_timestamp_train_{P}_with_masks`
    - `mix_timestamp_train_{P}_without_masks` (using the 10% negative samples rule as mentioned before)
    - `unlabeled` (unlabeled data)

The placeholder `{P}` can be `100`, `80`, `60`, `40`, or `20`. For example, model `M-mix-100` uses `mix_timestamp_train_100_with_masks` (which means combining `expert_timestamp_train_100_with_masks` and `citizen_with_mask` datasets), `mix_timestamp_train_100_without_masks`, and `unlabeled` datasets ffor finetuning.

So, we have the `M-mix-100`, `M-mix-80`, `M-mix-60`, `M-mix-40`, `M-mix-20`, and a base model `M-citizen` (which can be considered `M-mix-0`) from the previous research question for comparison. By doing so, we know the effect of adding various levels of expert contribution. We anticipate that `M-mix-100` will be our best model, which serves as the upper bound of performance.

### Ablation studies

Notice that in this experiment setting for RQ1 and RQ2, we have an assumption that using unlabeled data can increase performance. If time permits, we should do an ablation study on the effect of removing the unlabeled data from the models to see the effect.

### Validation and testing

All models (except `M-zeroshot`) will use the following validation set for model selection:
- `expert_timestamp_val_with_masks`
- `expert_timestamp_val_without_masks`

The `M-zeroshot` model should use the test set in the SMOKE5K dataset for validation.

All models will use the following test set for performance evaluation:
- `expert_timestamp_test_with_masks`
- `expert_timestamp_test_without_masks`

## Evaluation Metrics

All models are evaluated using the `evaluate_new` function (implemented in `unimatch_v2/supervised.py`), which computes the following metrics on the smoke segmentation task.

### Global metrics

Global metrics are computed by accumulating true positives (TP), false positives (FP), and false negatives (FN) across all pixels in the entire test set, treating every pixel equally regardless of which image it belongs to.

- **gIoU** (Global Intersection over Union): $\frac{TP}{TP + FP + FN}$. Measures the overlap between the predicted and ground-truth smoke regions across all images.
- **gF1** (Global F1 Score): $\frac{2 \cdot gPre \cdot gRec}{gPre + gRec}$. Harmonic mean of global precision and recall.
- **gPre** (Global Precision): $\frac{TP}{TP + FP}$. Fraction of predicted smoke pixels that are actually smoke.
- **gRec** (Global Recall): $\frac{TP}{TP + FN}$. Fraction of actual smoke pixels that are correctly predicted.
- **gAccu** (Global Accuracy): $\frac{\text{correct pixels}}{\text{total pixels}}$. Overall pixel-level classification accuracy across all images.

### Per-image metrics

Per-image metrics are computed per positive image (i.e., images where the ground truth contains at least one smoke pixel) and then averaged. Negative images (no smoke in ground truth) are excluded from these metrics to avoid degenerate scores.

- **mIoU** (Mean IoU): average IoU computed per positive image.
- **mF1** (Mean F1): average F1 score computed per positive image.

### False alarm metric

- **FAR** (False Alarm Rate): fraction of negative images (no smoke in ground truth) where the model incorrectly predicts a cluster of more than 10 smoke pixels. A lower FAR means fewer false detections on clean images.

### Model selection criterion

During training, the model checkpoint with the best `gF1` on the validation set is selected as the final model for testing.
