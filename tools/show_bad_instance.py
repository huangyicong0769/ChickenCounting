import argparse
import random
import string
import time
import os
import queue
import cv2
import numpy as np
import shutil
from tqdm import tqdm

from typing import List, Dict, Optional, Union
from ultralytics import YOLO

def generate_safe_dirname(seed=None):
    if seed is None:
        seed = int(time.time())
    random.seed(seed)
    print(f"Using seed: {seed}")
    # Use letters, numbers and underscores only to ensure compatibility
    safe_chars = string.ascii_letters + string.digits
    # Generate a 16-character string
    return ''.join(random.choice(safe_chars) for _ in range(16))

def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes in XYWH format.
    
    Args:
        bbox1: [center_x, center_y, width, height]
        bbox2: [center_x, center_y, width, height]
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Convert XYWH to XYXY format (top-left and bottom-right corners)
    # For bbox1
    x1_min = bbox1[0] - bbox1[2]/2
    y1_min = bbox1[1] - bbox1[3]/2
    x1_max = bbox1[0] + bbox1[2]/2
    y1_max = bbox1[1] + bbox1[3]/2
    
    # For bbox2
    x2_min = bbox2[0] - bbox2[2]/2
    y2_min = bbox2[1] - bbox2[3]/2
    x2_max = bbox2[0] + bbox2[2]/2
    y2_max = bbox2[1] + bbox2[3]/2

    # Get coordinates of intersection rectangle
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    # Check if there is no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both bounding boxes
    bbox1_area = bbox1[2] * bbox1[3]  # width * height for first box
    bbox2_area = bbox2[2] * bbox2[3]  # width * height for second box

    # Calculate union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def parse_label_file(lbl_dir, img_w=None, img_h=None):
    """
    Parse a label text file where each line contains:
    class_id center_x center_y width height
    
    Args:
        lbl_dir: Path to the label text file
        img_w: Width of the image (required for denormalization)
        img_h: Height of the image (required for denormalization)
        
    Returns:
        list: List of dictionaries containing parsed data for each line
    """
    try:
        with open(lbl_dir, 'r') as file:
            lines = file.readlines()
            
        bboxes, clss = [], []
        for line_num, line in enumerate(lines, 1):
            try:
                # Split the line and convert to appropriate types
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Warning: Line {line_num} has incorrect number of elements")
                    continue
                
                # Convert to appropriate types
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Perform normalization/denormalization if requested
                if (img_w is not None) and (img_h is not None):
                    # Convert normalized coordinates to absolute pixels
                    center_x = center_x * img_w
                    center_y = center_y * img_h
                    width = width * img_w
                    height = height * img_h

                bboxes.append([center_x, center_y, width, height])
                clss.append(class_id)
                
            except ValueError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
                
        return bboxes, clss
    
    except FileNotFoundError:
        print(f"Error: File not found at {lbl_dir}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def visualize(bbox_list, img, save_path):
    """
    Visualize and save images with bounding boxes, considering different scenarios and transparency.
    
    Args:
        bbox_list: List of dictionaries containing bbox, gbbox, cls, iou, and conf information
        img: Path to the input image
        save_path: Path where the output image should be saved
    """
    # Check if input image exists
    if not os.path.exists(img):
        raise FileNotFoundError(f"Input image not found at {img}")

    # Read the image
    image = cv2.imread(img)
    if image is None:
        raise ValueError("Failed to load image")
    
    # Create a copy for overlay
    backup = image.copy()
    all = image.copy()
    
    # Color definitions (BGR format)
    COLORS = {
        'both': (255, 127, 123), # Soft Coral
        'bbox_only': (204, 204, 255), # Periwinkle for bbox only
        'gbbox_only': (152, 255, 178), # Mint Green for gbbox only
        'intersection': (127, 127, 127), # Gray for intersection
        'difference': (255, 218, 185), # Peach for difference
    }
    
    def convert_bbox(bbox: List) -> tuple:
        """Convert center coordinates to top-left and bottom-right coordinates"""
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        return (x1, y1, x2, y2)
    
    def draw_bbox(img: np.ndarray, bbox: tuple, color: tuple, label: str|None = None, thickness: int = 2) -> None:
        """Draw a bounding box on the image"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        if label is not None:
            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
            # Draw background rectangle for text
            padding = 5
            cv2.rectangle(img, 
                          (x1, y1 - text_height - 2*padding),
                          (x1 + text_width + 2*padding, y1),
                          color, -1)  # -1 fills the rectangle
        
            # Draw text
            cv2.putText(img, label, 
                        (x1 + padding, y1 - padding), 
                        font, font_scale, (255, 255, 255), 
                        font_thickness)
    
    def create_mask(shape: tuple, bbox: tuple) -> np.ndarray:
        """Create a binary mask for a bounding box"""
        mask = np.zeros(shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return mask
    
    # Process each bbox entry
    for i, entry in enumerate(bbox_list):
        image = backup.copy()

        # Calculate transparency based on confidence
        conf = entry.get('conf', None)
        alpha = 0.4 if conf is None else conf-0.5  # Clamp between 0.3 and 0.5
        
        if entry.get('bbox') and entry.get('gbbox'):
            # Both boxes exist - draw intersection and difference
            bbox_coords = convert_bbox(entry['bbox'])
            gbbox_coords = convert_bbox(entry['gbbox'])
            
            # Create masks for both boxes
            bbox_mask = create_mask(image.shape, bbox_coords)
            gbbox_mask = create_mask(image.shape, gbbox_coords)
            
            # Calculate intersection and difference masks
            # intersection_mask = cv2.bitwise_and(bbox_mask, gbbox_mask)
            difference_mask = cv2.bitwise_xor(bbox_mask, gbbox_mask)
            
            # Draw original boxes
            draw_bbox(image, bbox_coords, COLORS['both'], label=f"{int(entry['cls'])}, {entry['conf']:3f}")
            draw_bbox(image, gbbox_coords, COLORS['both'])
            draw_bbox(all, bbox_coords, COLORS['both'], label=f"{i}")
            draw_bbox(all, gbbox_coords, COLORS['both'])
            
            # Apply colored overlays for intersection and difference
            overlay_color = image.copy()
            all_color = all.copy()
            # image_color[intersection_mask > 0] = COLORS['intersection']
            overlay_color[difference_mask > 0] = COLORS['difference']
            all_color[difference_mask > 0] = COLORS['difference']
            
            # Blend with original image
            cv2.addWeighted(overlay_color, alpha, image, 1 - alpha, 0, image)
            cv2.addWeighted(all_color, alpha, all, 1 - alpha, 0, all)
            
        elif entry.get('bbox'):
            # Only bbox exists
            bbox_coords = convert_bbox(entry['bbox'])
            draw_bbox(image, bbox_coords, COLORS['bbox_only'], label=f"{int(entry['cls'])}, {entry['conf']:3f}")
            draw_bbox(all, bbox_coords, COLORS['bbox_only'], label=f"{i}")
            
            # Create and apply overlay
            mask = create_mask(image.shape, bbox_coords)
            overlay_color = image.copy()
            all_color = all.copy()
            overlay_color[mask > 0] = COLORS['bbox_only']
            all_color[mask > 0] = COLORS['bbox_only']
            cv2.addWeighted(overlay_color, alpha, image, 1 - alpha, 0, image)
            cv2.addWeighted(all_color, alpha, all, 1 - alpha, 0, all)
            
        elif entry.get('gbbox'):
            # Only gbbox exists
            gbbox_coords = convert_bbox(entry['gbbox'])
            draw_bbox(image, gbbox_coords, COLORS['gbbox_only'], label=f"{int(entry['cls'])}")
            draw_bbox(all, gbbox_coords, COLORS['gbbox_only'], label=f"{i}")
            
            # Create and apply overlay
            mask = create_mask(image.shape, gbbox_coords)
            overlay_color = image.copy()
            all_color = all.copy()
            overlay_color[mask > 0] = COLORS['gbbox_only']
            all_color[mask > 0] = COLORS['gbbox_only']
            cv2.addWeighted(overlay_color, alpha, image, 1 - alpha, 0, image)
            cv2.addWeighted(all_color, alpha, all, 1 - alpha, 0, all)
        else:
            continue
    
        # Save the final image
        save_file = os.path.join(save_path, f"{i}.jpg")
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        # print(save_file)
        cv2.imwrite(save_file, image)
        
    save_file = os.path.join(save_path, f"all.jpg")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    cv2.imwrite(save_file, all)

def main(args):
    model = YOLO(args.weight)
    img_dir = args.img_dir
    assert os.path.basename(img_dir) == 'images'
    lbl_dir = os.path.join(os.path.dirname(img_dir), 'labels')
    q = queue.PriorityQueue()

    # Clear and recreate the save directory
    if os.path.exists(args.output):
        try:
            shutil.rmtree(args.output)
        except Exception as e:
            raise Exception(f"Failed to clear directory {args.output}: {str(e)}")
    
    # Create fresh output directory
    try:
        os.makedirs(args.output)
    except Exception as e:
        raise Exception(f"Failed to create directory {args.output}: {str(e)}")

    for img in tqdm(os.listdir(img_dir)):
        if not img.endswith(args.img_ext):
            continue
        lbl = os.path.join(lbl_dir, os.path.splitext(img)[0] + '.txt')
        out = os.path.join(args.output, os.path.splitext(img)[0])
        img = os.path.join(img_dir, img)

        if not os.path.exists(lbl):
            print(f"Warning: Missing label for {img}")
            continue

        result = model(img, iou=args.iou_conf, augment=True, conf=float(args.conf))[0]
        tboxes, tclss = parse_label_file(lbl, img_h=result.orig_shape[0], img_w=result.orig_shape[1])
        bboxes, clss, confs = result.boxes.xywh.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()

        res = []
        err = 0
        for bbox, cls, conf in zip(bboxes, clss, confs):
            k = None
            max_iou = 0
            for i in range(len(tboxes)):
                if tclss[i] != cls:
                    continue
                iou = calculate_iou(tboxes[i], bbox)
                if iou > args.iou_conf and iou > max_iou:
                    max_iou = iou
                    k = i
            if k is not None:
                res.append({'bbox':bbox, 'gbbox':tboxes[k], 'cls':cls, 'iou':max_iou, 'conf':conf})
                del tboxes[k]
                del tclss[k]
            else:
                res.append({'bbox':bbox, 'gbbox':None, 'cls':cls, 'iou':None, 'conf':conf})
                err += 1
                pass
        
        for tbox, tcls in zip(tboxes, tclss):
            res.append({'bbox':None, 'gbbox':tbox, 'cls':tcls, 'iou':None, 'conf':None})
            err += 1
            
        # print(f"img:{img}\n{'\n'.join(map(str, res))}")
        visualize(res, img, save_path=out+f"_Error{err}")
        # break
        q.put((err*-1, img))
    
    print("Top error image:")
    for i in range(10):
        if q.empty():
            break
        err, img = q.get()
        print(f"{img} has {err*-1} errors")
        

if __name__ == "__main__":
    default_output_dir = '.'+generate_safe_dirname()
    parser = argparse.ArgumentParser(description="Predict on specific image given model")
    parser.add_argument("img_dir", help="directory of images")
    parser.add_argument("-w", "--weight", required=True, help="directory of model weight")
    parser.add_argument("-c", "--conf", default=0.25, help="Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.")
    parser.add_argument("-o", "--output", default=f'./{default_output_dir}', help="directory of result")
    parser.add_argument("-e", "--img_ext", default=".jpg", help="images extention")
    parser.add_argument("-i", "--iou_conf", default=0.7, help="Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.")
    main(args=parser.parse_args())