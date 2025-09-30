python smoke_dataset.py dataset/smoke5k/test/test.txt dataset/smoke5k/test/ smoke5k_test
python smoke_dataset.py dataset/smoke5k/train/train.txt dataset/smoke5k/train/ smoke5k_train
python smoke_dataset.py dataset/ijmond_pseudo_masks/train_with_mask.txt dataset/ijmond_pseudo_masks/ ijmond_pseudo_mask_with_mask
python smoke_dataset.py dataset/ijmond_pseudo_masks/train_without_mask.txt dataset/ijmond_pseudo_masks/ ijmond_pseudo_mask_without_mask
python smoke_dataset.py dataset/ijmond_vid/unlabeled.txt dataset/ijmond_vid/ ijmond_vid_unlabeled
python smoke_dataset.py dataset/ijmond_seg/test/cropped/test_with_mask.txt dataset/ijmond_seg/test/cropped/ ijmond_seg_cropped_with_mask
python smoke_dataset.py dataset/ijmond_seg/test/cropped/test_without_mask.txt dataset/ijmond_seg/test/cropped/ ijmond_seg_cropped_without_mask