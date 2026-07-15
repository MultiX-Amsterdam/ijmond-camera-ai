import cv2
import os
import argparse


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, default="videos")
    parser.add_argument('--output_folder', type=str, default="frames")
    opt = parser.parse_args()
    return opt


def extract_frames(video_path, output_folder, all_txt_entries):
    """
    Extract frames from a video file and save them as .png images.
    Collects txt entries for a single combined txt file.
    Args:
        video_path: path to the video file
        output_folder: path to the folder where the frames will be saved
        all_txt_entries: list to collect all txt entries across videos
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)

    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    # Create img subdirectory
    img_folder = os.path.join(video_output_folder, "img")
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Read and save frames
    frame_count = 0
    print(f"Extracting frames from {video_path}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Save frame as .png file
        frame_filename = f"img_{frame_count:04d}.png"
        frame_path = os.path.join(img_folder, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Add entry to global txt file list (include video folder in path)
        all_txt_entries.append(f"{output_folder}/{video_name}/img/{frame_filename} None")

    print(f"Extracted {frame_count} frames to {video_output_folder}")

    # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = arg_parse()

    # Create output folder if it doesn't exist
    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    # Collect all txt entries from all videos
    all_txt_entries = []

    # Extract frames from each video
    for vid in os.listdir(opt.video_folder):
        if vid.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')):  # Only process video files
            extract_frames(os.path.join(opt.video_folder, vid), opt.output_folder, all_txt_entries)

    # Write single txt file with all entries
    txt_filename = "unlabeled.txt"
    with open(txt_filename, 'w') as f:
        for entry in all_txt_entries:
            f.write(f"{entry}\n")

    print(f"Created combined txt file: {txt_filename} with {len(all_txt_entries)} entries")