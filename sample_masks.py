from sklearn.cluster import KMeans
import os
import numpy as np
from operator import itemgetter
from utils import create_sub_images, load_json, load_pickle
import cv2
import json
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="data")
    parser.add_argument('--mask_folder', type=str, default="masks")
    parser.add_argument('--input_folder', type=str, default="frames")
    parser.add_argument('--output_folder', type=str, default="bbox")
    parser.add_argument('--store_bbox', type=bool, default=True)
    parser.add_argument('--feature_folder', type=str, default='data/features')
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--num_elements', type=int, default=3, help="Number of elements to select from each cluster")
    parser.add_argument('--video_metadata', type=str, default='metadata_ijmond_jan_22_2024.json')
    opt = parser.parse_args()
    return opt

def labels_selection(selected_frames, mask_folder_fullpath, input_folder_fullpath, output_folder, store_bbox, vid):
    """
    This function that receives a list of selected frames and creates sub-images for each frame. It also stores the bounding boxes in a json file.
    Args:
        selected_frames: A list of selected frames.
        mask_folder_fullpath: The full path to the mask folder.
        input_folder_fullpath: The full path to the input folder.
        output_folder: The full path to the output folder.
        store_bbox: A boolean to store the bounding boxes.
        vid: The video name.
    Returns:    
        coordinates: A list containing the metadata of the sub-images.
        num_sum_images: The number of sub-images.
    """
    num_sum_images = 0
    for file in selected_frames: 
        frame_num = int(file.split(".")[0].split("_")[-1])
        image_rgb = cv2.imread(os.path.join(input_folder_fullpath, vid, file.split(".")[0] + ".jpg"))
        sub_images, sub_images_with_bbox, bbox, sub_masks = create_sub_images(os.path.join(mask_folder_fullpath, vid, file), image_rgb, file.split(".")[0])

        for i, sub_image in enumerate(sub_images):
            # store the cropped frame
            output_vid_folder = os.path.join(output_folder, vid, str(frame_num), vid + "-" + str(frame_num) + "-" + str(i))
            if not os.path.exists(output_vid_folder):
                os.makedirs(output_vid_folder)
            cv2.imwrite(os.path.join(output_vid_folder, "crop.png"), sub_image)
            # store the cropped mask
            cv2.imwrite(os.path.join(output_vid_folder, "mask.png"), sub_masks[i])
            if store_bbox:
                cv2.imwrite(os.path.join(output_vid_folder, "crop_with_bbox.png"), sub_images_with_bbox[i])
            # store the metadata
            with open(os.path.join(output_vid_folder, "metadata.json"), "w") as f:
                json.dump(bbox[i], f)

        num_sum_images += len(sub_images)
    return num_sum_images


if __name__ == '__main__':
    opt = args_parser()
    data_folder = opt.data_folder
    mask_folder = opt.mask_folder
    input_folder = opt.input_folder
    output_folder = opt.output_folder
    store_bbox = opt.store_bbox
    mask_folder_fullpath = os.path.join(data_folder, mask_folder)
    input_folder_fullpath = os.path.join(data_folder, input_folder)
    feature_folder = opt.feature_folder
    num_clusters = opt.num_clusters
    num_elements = opt.num_elements
    total_samples = 0

    video_metadata = load_json(opt.video_metadata)
    
    # video_names = ["_1jFnujWn50-0", "zl6ckY2YM8c-2", "zOt2vMuYLx4-0"]
    video_lookup = {v["file_name"]: v for v in video_metadata}

    for vid in os.listdir(feature_folder):
        # For every video, we will select the closest elements to the clusters' center
        # if vid.split('_output')[0] not in video_names:
        #     continue
        print(vid.split('_output')[0])
        vid_path = os.path.join(feature_folder, vid)
        features = load_pickle(vid_path)
        
        # Apply KMeans to cluster the features
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        features_values = [v for k, v in features.items()]
        features_values = np.array(features_values).reshape(len(features),-1)
        kmeans.fit(features_values)
        cluster_centers = kmeans.cluster_centers_

        # Select the closest feature to the cluster center
        total_selected_elements = []
        for i, center in enumerate(cluster_centers):
            distances = np.linalg.norm(features_values - center, axis=1)
            # Take the indices of the 3 smallest distances
            idxes = np.argsort(distances)
            getter = itemgetter(*idxes[:num_elements])
            selected_elements = getter(list(features.keys()))
            total_selected_elements.extend(selected_elements)

        num_sum_images = labels_selection(total_selected_elements, mask_folder_fullpath, input_folder_fullpath, output_folder, store_bbox, vid.split('_output')[0])
        
        # Store the metadata of the video in a json file
        with open(os.path.join(output_folder, vid.split('_output')[0], "video.json"), "w") as f:
            json.dump(video_lookup[vid.split('_output')[0]], f)
        
        total_samples += num_sum_images
    
    print(f"Total samples: {total_samples}")
        
