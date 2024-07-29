import cv2
import numpy as np
import json
import pickle

def load_json(file_path):
    """
    This function loads a dictionary from a json file.
    Args:
        file_path: The path to the json file.
    Returns:
        The dictionary stored in the json file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_pickle(file_path):
    """
    This function loads a dictionary from a pickle file.
    Args:
        file_path: The path to the pickle file.
    Returns:
        The dictionary stored in the pickle file.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return  data 

def find_cropped_bbox(x, w, size , image_size):
    """
    Finds the cropped bounding box.
    Args:
        x: The x coordinate of the bounding box.
        w: The width of the bounding box.
        size: The size of the cropped image.
        image_size: The size of the image.
    Returns:
        The min and max x coordinates of the cropped bounding box.
    """
    offset_min = -1*(x + w/2 - size/2) if x + w/2 - size/2 < 0 else 0
    offset_max = x + w/2 + size/2 - image_size if x + w/2 + size/2 > image_size else 0
    min_x = max(0, x + w/2 - size/2 - offset_max)
    max_x = min(image_size, x + w/2 + size/2 + offset_min)
    assert int(max_x) - int(min_x) == size 
    return int(min_x), int(max_x)

def create_sub_images(mask_path, image_rgb, frame):
    """
    Creates sub-images based on the white regions from a binary mask.
    Args:
        mask_path: the path for the mask
        image_rgb: the image in RGB format
        frame: the frame names
    Returns:
        A list of sub-images containing the white regions from the mask.
    """
    bn_mask = cv2.imread(mask_path)
    bn_mask = cv2.cvtColor(bn_mask, cv2.COLOR_BGR2GRAY)
    # bn_mask[bn_mask<10] = 0
    _, mask = cv2.threshold(bn_mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    size = 512
    sub_images = []
    sub_images_with_bbox = []
    sub_masks = []
    metadata = []

    # Iterate through contours
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # box = [x, y, w, h]

        # Skip if the bounding box is too big
        if w >= 512 or h >= 512:
            continue
        # Skip if the bounding box is too small
        if w <= 10 or h <= 10:
            continue

        # Cut a rectangle in the image sized 512x512 thats includes the bbox
        min_x, max_x = find_cropped_bbox(x, w, size, image_rgb.shape[1])
        min_y, max_y = find_cropped_bbox(y, h, size, image_rgb.shape[0])
        sub_image = image_rgb[min_y:max_y, min_x:max_x]
        # Crop the mask
        mask_temp = np.zeros_like(mask)
        sub_mask = (bn_mask * cv2.drawContours(mask_temp, [cnt], -1, (1, 1, 1), cv2.FILLED))[min_y:max_y, min_x:max_x]

        relative_x = x - min_x
        relative_y = y - min_y
        temp_img = sub_image.copy()
        cv2.rectangle(temp_img, (relative_x, relative_y), (relative_x + w, relative_y + h), (0, 255, 0), 2)
        # relative_box = [relative_x, relative_y, w, h]
        
        sub_images.append(sub_image)
        sub_images_with_bbox.append(temp_img)
        sub_masks.append(sub_mask)
        
        metadata.append({'boxes': {'x': x, 'y': y, 'w': w, 'h': h}, 
                        "relative_boxes": {'x': relative_x, 'y': relative_y, 'w': w, 'h': h},
                        "image_width": bn_mask.shape[0], 
                        "image_height": bn_mask.shape[1], 
                        "cropped_width": size, 
                        "cropped_height": size})

    # coordinates[frame] = {"relative_boxes": relative_boxes, "boxes": bboxes, "image_width": image_rgb.shape[1], "image_height": image_rgb.shape[0], "cropped_size": size}
    
    return sub_images, sub_images_with_bbox, metadata, sub_masks
