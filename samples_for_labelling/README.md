# ijmond-camera-ai

### Table of Content
- [Download the videos](#download-videos)
- [Extract frames](#extract-frames)
- [Inference the BVM model](#inference-bvm)
- [Sample masks for labelling](#sample-masks)


## <a name="download-videos"></a>Download videos
The videos to be downloaded are identified by codes 23 and 47, and there should be a total of 481 videos. Each video contains 36 frames. The videos are stored in the "./data/videos" folder.
```
python download_videos.py
```
Structure of files system so far:
```
└── data # the root folder
    └── videos # the folder that contains all the downloaded videos
        ├── _1jFnujWn50-0.mp4 # example of video
        └── ... 
```

## <a name="extract-frames"></a>Extract frames
In order to process videos and extract the segmentation masks, the frames need to be extracted. This is done using the line below. video_folder is the folder containing all the videos, and output_folder is where the extracted frames will be stored.
```
python preprocessing.py --video_folder data/videos --output_folder data/frames
```

Structure of files system so far:
```
└── data # the root folder
    └── videos # the folder that contains all the downloaded videos
        ├── _1jFnujWn50-0.mp4 # example of video
        └── ... 
    └── frames # the folder that contains the frames of the videos
        ├── _1jFnujWn50-0 # video folder
            ├── frame_0001.jpg # frame 1
            └── ...  
        └── ... # video folders
    
```

## <a name="inference-bvm"></a>Inference the BVM model
The trans_bvm folder contains the [`BVM`](https://github.com/SiyuanYan1/Transmission-BVM/tree/main)  model. To run the model, you need to specify the input directory frames_folder where the frames of each video are stored, and the output directory output_folder where all the masks and features for each frame of the video will be stored. The features will be used in the frame selection phase to select the most representative frames for labeling from each video. If you do not want to store the features, simply remove the --mode_features option from the command below.

At the end, two new folders will be created in the data folder: features, which will include the features of the frames of the videos, and masks, which will contain all the masks found by the BVM.

```
python trans_bvm/test.py --frames_folder data/frames --output_folder data --mode_features
```

Structure of files system so far:
```
└── data # the root folder
    └── videos # the folder that contains all the downloaded videos
        ├── _1jFnujWn50-0.mp4 # example of video
        └── ... 
    └── frames # the folder that contains the frames of the videos
        ├── _1jFnujWn50-0 # video folder
            ├── frame_0001.jpg # frame 1
            └── ...  
        └── ... # video folders
    └── features # the folder that contains all the feature files
        ├── _1jFnujWn50-0_output.pkl # example of feature file
        └── ... 
    └── masks # the folder that contains the masks of the videos
        ├── _1jFnujWn50-0 # video folder
            ├── frame_0001.jpg # mask for frame 1
            └── ...  
        └── ... # video folders
```


## <a name="sample-masks"></a>Sample masks for labelling
In order to select the most representative frames for labeling, a clustering algorithm was used. The algorithm employed is KMeans, with a default of 5 clusters, which can be adjusted through the arguments. The features for each frame were used for the clustering process. The selected frames are the top 3  closest to the center of each cluster, resulting in 15 frames in total. After selecting the frames, they are cropped to a region close to the bounding box of the segmentation mask. Once the process is complete, all the data are stored in the bbox folder.

The number of total representative frames is: 11681

```
python sample_masks.py --data_folder data --output_folder bbox --num_clusters 5 --num_elements 3
```
Structure of files system:
```
└── bbox # the root folder that stores all bounding boxes
    └── A9W8G55JucU3 # unique ID that represents the video, can be the video file name (without extension)
        ├── video.json # the video metadata
        └── 0 # frame number
            └── A9W8G55JucU3-0-0 # unique ID that represents the cropped region in the frame
                ├── mask.png # the grayscale mask (created by the segmentation model)
                ├── crop.png # the cropped image in the video frame
                └── metadata.json # storing the bounding box information
```
Structure in the metadata.json file:
```sh
{
    "boxes": [x, y, w, h],  # bounding box based on the original video frame (not cropped)
    "relative_boxes": [x, y, w, h], # bounding box based on the cropped region
    "image_width": w, # width of the original video frame
    "image_height": h, # height of the original video frame
    "cropped_width": cw, # width of the cropped region
    "cropped_height": ch, # height of the cropped region
}
```
See the `example_output` folder for examples.
