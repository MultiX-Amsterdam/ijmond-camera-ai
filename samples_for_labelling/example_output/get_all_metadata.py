"""
This file will loop all the subfolders in the "bbox_batch_1" folder,
and then output a json file that contain metadata for all segmentation masks.

The combined metadata will be used for the following application:
(for entering the metadata in to the database)
    - https://github.com/MultiX-Amsterdam/ijmond-camera-monitor

The "bbox_batch_1" contains the output from the samples_for_labelling tool.
"""

import os
import json

# Root folder where the data is stored
root_folder = "bbox_batch_1"

# List to store combined data
combined_data = []

# Loop through the folder structure
for video_folder in os.listdir(root_folder):
    video_folder_path = os.path.join(root_folder, video_folder)

    # Load video.json file
    video_json_path = os.path.join(video_folder_path, 'video.json')
    if not os.path.exists(video_json_path):
        continue

    with open(video_json_path, 'r') as f:
        video_data = json.load(f)

    # Extract video metadata from video.json
    video_id = video_data.get('id')
    video_timestamp = video_data.get('start_time')

    # Loop through frame folders inside each video folder
    for frame_folder in os.listdir(video_folder_path):
        frame_folder_path = os.path.join(video_folder_path, frame_folder)

        # Check if the frame folder is a directory
        if not os.path.isdir(frame_folder_path):
            continue

        # Loop through mask folders inside each frame folder
        for mask_folder in os.listdir(frame_folder_path):
            mask_folder_path = os.path.join(frame_folder_path, mask_folder)

            # Check if the mask folder is a directory
            if not os.path.isdir(mask_folder_path):
                continue

            # Path to metadata.json file
            metadata_json_path = os.path.join(mask_folder_path, 'metadata.json')

            # Check if metadata.json exists
            if os.path.exists(metadata_json_path):
                with open(metadata_json_path, 'r') as f:
                    metadata = json.load(f)

                # Prepare the combined data entry
                # Note that the frame_timestamp here uses the one from the video
                data_entry = {
                    "mask_file_name": "mask.png",
                    "image_file_name": "crop.png",
                    "file_path": mask_folder_path + '/',  # Ensure the path ends with a slash
                    "frame_number": int(frame_folder),
                    "frame_timestamp": int(video_timestamp),
                    "video_id": video_id,
                    "relative_boxes": metadata.get("relative_boxes"),
                    "cropped_width": metadata.get("cropped_width"),
                    "cropped_height": metadata.get("cropped_height")
                }

                # Append the entry to the list
                combined_data.append(data_entry)

# Save combined data to a JSON file
output_file = "combined_metadata.json"
with open(output_file, 'w') as f:
    json.dump(combined_data, f, indent=4)

print(f"Combined metadata has been written to {output_file}")
