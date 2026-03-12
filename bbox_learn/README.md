# Weakly Semi-Supervised Learning

This folder contains a weakly semi-supervised learning pipeline that uses bounding boxes and unlabeled images (highly likely to contain smoke emissions) to train a segmentation model, which outputs pixel-level masks for industrial smoke segmentation. The segmentation model is first pretrained on a dataset from another context (a combination of wildfire detection and synthetic smoke) and then fine-tuned in our context using bounding boxes and unlabeled smoke images.

## Data Preparation

### Prepare citizen-labeled IJmond bounding boxes

Prepare IJmond bounding boxes for training. This will download images to the `dataset/ijmond_bbox/img` folder, create a `filtered_bbox_labels_1_aug_2025.json` file with filtered and aggregated bounding boxes, and create `.npy` files in the `dataset/ijmond_bbox/img_npy` folder, and create debugging images to the `dataset/ijmond_bbox/debug` folder.
```sh
python download_ijmond_bbox_images.py dataset/ijmond_bbox/bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img
python filter_aggr_bbox_and_create_npy.py dataset/ijmond_bbox/bbox_labels_1_aug_2025.json dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json dataset/ijmond_bbox/
```

Test if the IJmond bounding boxes can be loaded. This will create a `debug_plot_ijmondbox.png` file for debugging.
```sh
python ijmond_bbox_dataset.py dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img_npy/
```

### Prepare SMOKE5K data for pretraining

Prepare SMOKE5K for training. This will create `.npy` files and also metadata txt files in `dataset/smoke5k/`.
```sh
python create_smoke5k_metadata_and_npy.py dataset/smoke5k/
```

Check if the SMOKE5K dataset can be loaded. This will create `debug_plot_smoke5k_test.png` and `debug_plot_smoke5k_test_transformed.png` files for debugging.
```sh
python smoke_dataset.py dataset/smoke5k/test/test.txt dataset/smoke5k/test/ smoke5k_test
```

### Create IJmond pseudo masks based on the bounding boxes

We use Segment Anything (SAM) to create the pseudo masks. Before doing this, run the following on the terminal to install SAM. You need to be in the `bbox_learn` directory.
```sh
pip install git+https://github.com/facebookresearch/segment-anything.git
wget -P https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
```

Create pseudo masks and metadata txt files (one with masks, one without masks) using the IJmond bounding boxes and save the masks in the `dataset/ijmond_pseudo_masks/` path. This will create `debug_plot_pseudo_masks.png` file for debugging.
```sh
python create_pseudo_masks.py dataset/ijmond_bbox/filtered_bbox_labels_1_aug_2025.json dataset/ijmond_bbox/img_npy/
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
        └── images # all camera images
            ├── XXX.jpg
            └── ...
        └── _annotations.coco.json # the annotation file
```

Then, run a script to crop the images. This will first create segmentation masks (under `dataset/ijmond_seg/test/masks/`) and then crop the large panoramas into smaller ones (under `dataset/ijmond_seg/test/cropped/`).
```sh
python create_ijmond_seg_masks.py dataset/ijmond_seg/test/images/ dataset/ijmond_seg/test/_annotations.coco.json dataset/ijmond_seg/test/masks/
python crop_ijmond_seg_and_create_npy.py dataset/ijmond_seg/test/images/ dataset/ijmond_seg/test/_annotations.coco.json dataset/ijmond_seg/test/masks/ dataset/ijmond_seg/test/cropped/
```

So, after that, the file structure should look like below:
```sh
└── dataset # the root folder
    └── ijmond_seg # the folder that contains the IJmond segmentation dataset
        └── cropped # all cropped images and masks
            └── images # all cropped camera images
                ├── XXX.jpg
                └── ...
            └── images_npy # all cropped camera images in numpy format
                ├── XXX.npy
                └── ...
            └── masks # all cropped masks
                ├── XXX.png
                └── ...
            └── masks # all cropped masks in numpy format
                ├── XXX.png
                └── ...
            └── test_with_mask.txt # paths for image-mask pairs (with masks)
            └── test_without_mask.txt # paths for image-mask pairs (with no masks)
            └── metadata.json # metadata for each cropped image
        └── images # all camera images
            ├── XXX.jpg
            └── ...
        └── masks # all masks
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
python smoke_dataset.py dataset/ijmond_seg/test/cropped/splits/split_by_timestamp/train/20_with_masks.txt ijmond_seg_cropped_train_with_mask_20
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
- `M-unsupervised`: finetuned using only unlabeled dataset `unlabeled`
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

All models will use the following test set for performance evaluation:
- `expert_timestamp_test_with_masks`
- `expert_timestamp_test_without_masks`

We now describe the evaluation metrics for the experiments, which applies to both validation and test sets. In general, we want to design metrics that are suitable for citizen science. In the citizen science context, we have many community members with various levels of expertises and motivations to help us find smoke emissions that may need further inspections. So our core philosophy is that this model will be embedded in a human-in-the-loop system where humans can check its outputs frequently. This means that the model's role is to help people quickly filter smoke emission events. In this sense, we focus on getting a high recall (rather than high precision) since we do not want to miss events. This means that we will use the `F2-score` metric when computing pixel-level performance, which prioritize the recall, rather than `F1-score` that weights precision and recall evenly.

Also, we focus more on the images with smoke (positive images) rather than negative images (i.e., the ones with no smoke). The rationality is that the model can be combined with a good image classification model that can remove many images with no smoke. In other words, the model is not meant to be used alone and should be integrated into a pipeline with many types of machine learning models. So in practice, in the community citizen science context, there is little need to pay attention on the images with no smoke. In this sense, we will use a higher weight for positive images and lower weight for negative images when computing the `IoU` metric (intersection over union) to evaluate the quality of the masks. We also only compute the `IoU` for smoke regions and not background regions to reflect our core philosophy of focusing on the performance of smoke plumes rather than the background.

We use both the pixel-level `F2-score` metric and geometric-based `IoU` metric together. The `F2-score` metric gives us information about if smoke regions are well covered, and the `IoU` metric shows the quality of the mask (i.e., if the mask aligns well with the smoke). This combination gives us several advantages to diagnose models. In the best scenario, when the predicted mask aligns well with the ground truth smoke, both metrics will have a high value. In the worst scenario that the mask misses the ground truth, both metrics will have a low value. However, if the model is agreesive and produces a mask that covers a large region of the smoke but at the same time also generates a bunch of false positives (i.e., marking regions outside the ground truth smoke as smoke), we will get a slightly lower `F2-score` (because it considers less about false positives) and a much lower `IoU` (because the union now becomes large). This is fine for finding smoke because we see an obvious alert of smoke plume on the community side. But this is not a good model that can be used for estimating the amount of smoke (such as in the case that we want to correlate smoke and air quality sensing data).

Also, when calculating `F2-score` and `IoU`, we first compute these metrics for each image and then average them together, and we call then `mF2` and `mIoU`. Why not using all the pixels across images? We think that it is important to treat each image independently. For example, if there is a large plume that covers almost the entire image, using a global way of computing the metrics will propagate a lot of errors into the final metric just based on mistakes from this single image, which is not fair. Notice that `mF2` and `mIoU` are different from the traditional definition of `mean F2` and `mean IoU`, where the averages are about averaging the metics across different class labels. In our case, we only consider the ground truth smoke (i.e., pixel value 1) and not background labels (i.e., pixel value 0), and we do the average across images.

To reflect our design philosophy that we care more about postive images, we weight the results of positve and negative images differently. In other words, we calculate `mF2` and `mIoU` for both positve and negative images, resulting in four main metrics `mF2_smoke`, `mIoU_smoke`, `mF2_clear` (no smoke), and `mIoU_clear` (no smoke). We then calculate the final `mF2` by taking a weighted sum of `mF2_smoke` and `mF2_clear` (same logic for `mIoU_smoke`). We give positive images `mF2_smoke` more weight (`w_pos=0.8`) than negative images `mF2_clear` less weight (`w_neg=0.2`).

The following code is the implementation of the evaluation metrics. It is written in PyTorch syntax but may need adjustment for being used in the experiment pipeline. For model selection during validation, use only `mF2` and `mIoU`. First, pick several candidates (with different training epoch checkpoints) that have a good level of `mIoU`. Then, pick the one with the highest `mF2` from the candidates. This two stage filtering is designed to make sure that the model have a good mask quality and also a good recall (with also a small consideration of precision). For getting the final performance, report all the metrics that are returned from the function.

```python
def evaluate_new(model, dataloader, w_pos=0.8, w_neg=0.2, threshold=0.5, multiplier=None):
    """
    Calculates weighted mIoU, mF2, mRecall, and mPrecision.
    """
    model.eval()

    # Grouped storage for per-image metrics
    smoke_f2s, smoke_ious, smoke_recalls, smoke_precisions = [], [], [], []
    clear_f2s, clear_ious, clear_recalls, clear_precisions = [], [], [], []
    smoke_accu, clear_accu = [], []

    smooth = 1e-7

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images, masks = images.cuda(), masks.cuda()

            if multiplier is not None:
                ori_h, ori_w = images.shape[-2:]
                if multiplier == 512:
                    new_h, new_w = 512, 512
                else:
                    new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(ori_w / multiplier + 0.5) * multiplier

                images = F.interpolate(images, (new_h, new_w), mode='bilinear', align_corners=True)

            outputs = model(images)



            if multiplier is not None:
                outputs = F.interpolate(outputs, (ori_h, ori_w), mode='bilinear', align_corners=True)

            preds = (outputs > threshold).float()
            preds = preds.argmax(dim = 1)

            intersection, union, target = \
                intersectionAndUnion(preds.cpu().numpy(), masks.cpu().numpy(), 2, 255)

            # --- CASE 1: NEGATIVE SAMPLE (Ground Truth is Empty) ---
            if masks.cpu().sum() == 0:
                score = 1.0 if preds.cpu().sum() == 0 else 0.0
                clear_ious.append(score)
                correct_pixels = (preds.cpu() == masks.cpu()).sum().item()
                clear_accu.append(correct_pixels / preds.numel())
            # --- CASE 2: POSITIVE SAMPLE (Smoke Present) ---
            else:
                iou_class = (intersection[1].sum() + smooth) / (union[1].sum() + smooth)
                smoke_ious.append(iou_class)
                correct_pixels = (preds.cpu() == masks.cpu()).sum().item()
                smoke_accu.append(correct_pixels / preds.numel())

            for p, m in zip(preds, masks):
                # Pixel-level components
                tp = (p * m).sum().item()
                fp = (p * (1 - m)).sum().item()
                fn = ((1 - p) * m).sum().item()
                union = p.sum().item() + m.sum().item() - tp

                precision = (tp + smooth) / (tp + fp + smooth)
                recall = (tp + smooth) / (tp + fn + smooth)

                # --- CASE 1: NEGATIVE SAMPLE (Ground Truth is Empty) ---
                if m.sum() == 0:
                    score = 1.0 if p.sum() == 0 else 0.0
                    clear_f2s.append(score)
                    clear_recalls.append(score)
                    clear_precisions.append(score)

                # --- CASE 2: POSITIVE SAMPLE (Smoke Present) ---
                else:
                    # F2 Score
                    f2 = (5 * precision * recall) / ( 4 * precision + recall + smooth)
                    smoke_f2s.append(f2)

                    # Recall and Precision
                    smoke_recalls.append(recall)
                    smoke_precisions.append(precision)

    # 1. Calculate the raw means for both groups
    mF2_smoke = np.mean(smoke_f2s) if smoke_f2s else 0.0
    mIoU_smoke = np.mean(smoke_ious) if smoke_ious else 0.0
    mRec_smoke = np.mean(smoke_recalls) if smoke_recalls else 0.0
    mPre_smoke = np.mean(smoke_precisions) if smoke_precisions else 0.0
    mAccu_smoke = np.mean(smoke_accu) if smoke_accu else 0.0

    mF2_clear = np.mean(clear_f2s) if clear_f2s else 0.0
    mIoU_clear = np.mean(clear_ious) if clear_ious else 0.0
    mRec_clear = np.mean(clear_recalls) if clear_recalls else 0.0
    mPre_clear = np.mean(clear_precisions) if clear_precisions else 0.0
    mAccu_clear = np.mean(clear_accu) if clear_accu else 0.0

    # 2. Compute Weighted Final Metrics (The ones used for ranking)
    weight_sum = w_pos + w_neg

    results = {
        "mIoU": (w_pos * mIoU_smoke + w_neg * mIoU_clear) / weight_sum,
        "mF2":  (w_pos * mF2_smoke + w_neg * mF2_clear) / weight_sum,
        "mRec": (w_pos * mRec_smoke + w_neg * mRec_clear) / weight_sum,
        "mPre": (w_pos * mPre_smoke + w_neg * mPre_clear) / weight_sum,
        "mAccu": (w_pos * mAccu_smoke + w_neg * mAccu_clear) / weight_sum,
        "mF2_smoke": mF2_smoke,
        "mIoU_smoke": mIoU_smoke,
        "mRec_smoke": mRec_smoke,
        "mPre_smoke": mPre_smoke,
        "mAccu_smoke": mAccu_smoke,
        "mF2_clear": mF2_clear,
        "mIoU_clear": mIoU_clear,
        "mRec_clear": mRec_clear,
        "mPre_clear": mPre_clear,
        "mAccu_clear": mAccu_clear
    }

    return results
```
