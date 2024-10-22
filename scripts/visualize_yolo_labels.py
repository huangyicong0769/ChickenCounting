import cv2
import os
import argparse
import glob

# Define a list of class names (modify according to your dataset)
CLASS_NAMES = [
    "chicken",
    "column",
    "egg",

    # Add more classes as needed
]

# Define colors for different classes
COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
    # Add more colors as needed
]

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on images in a directory.")
    parser.add_argument(
        "--images_dir", "-i", required=True, help="Path to the directory containing images."
    )
    parser.add_argument(
        "--labels_dir", "-l", required=True, help="Path to the directory containing YOLO label files."
    )
    parser.add_argument(
        "--class_names",
        "-c",
        nargs='*',
        default=CLASS_NAMES,
        help="List of class names. Default classes will be used if not specified.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default=None,
        help="Path to save the output images with visualized labels. If not provided, images will be displayed interactively.",
    )
    parser.add_argument(
        "--image_extensions",
        "-e",
        nargs='*',
        default=['.jpg', '.jpeg', '.png', '.bmp'],
        help="List of image file extensions to consider. Default: ['.jpg', '.jpeg', '.png', '.bmp']",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action='store_true',
        help="Enable verbose output for debugging.",
    )
    return parser.parse_args()

def normalize_extensions(extensions):
    """
    Ensure all extensions start with a dot and are in lowercase.
    """
    normalized = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        normalized.append(ext.lower())
    return normalized

def load_labels(label_path, verbose=False):
    """
    Load YOLO labels from a file.
    Each line in the file should be: <class_id> <x_center> <y_center> <width> <height>
    """
    labels = []
    if not os.path.isfile(label_path):
        if verbose:
            print(f"Warning: Label file {label_path} does not exist.")
        return labels  # Return empty list; image may have no labels

    with open(label_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                if verbose:
                    print(f"Invalid label format in {label_path} at line {line_num}: {line.strip()}")
                continue
            try:
                class_id, x_center, y_center, width, height = map(float, parts)
                labels.append({
                    "class_id": int(class_id),
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height
                })
            except ValueError:
                if verbose:
                    print(f"Non-numeric label values in {label_path} at line {line_num}: {line.strip()}")
                continue
    return labels

def visualize(image, labels, class_names):
    """
    Draw bounding boxes on the image based on YOLO labels.
    """
    image_height, image_width = image.shape[:2]

    for label in labels:
        class_id = label["class_id"]
        x_center = label["x_center"] * image_width
        y_center = label["y_center"] * image_height
        width = label["width"] * image_width
        height = label["height"] * image_height

        # Calculate top-left and bottom-right coordinates
        x1 = int(max(x_center - width / 2, 0))
        y1 = int(max(y_center - height / 2, 0))
        x2 = int(min(x_center + width / 2, image_width - 1))
        y2 = int(min(y_center + height / 2, image_height - 1))

        # Choose color based on class_id
        color = COLORS[class_id % len(COLORS)]

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Put class name text
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class {class_id}"
        label_text = f"{class_name} ({class_id})"
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline),
            (x1 + text_width, y1),
            color,
            thickness=cv2.FILLED
        )
        cv2.putText(
            image,
            label_text,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    return image

def get_image_files(images_dir, image_extensions, verbose=False):
    """
    Retrieve a sorted list of unique image file paths from the specified directory
    matching the given extensions. Ensures no duplicates and handles case-insensitivity.
    """
    image_files = set()
    for ext in image_extensions:
        # Use case-insensitive pattern by including both lowercase and uppercase
        pattern_lower = os.path.join(images_dir, f"*{ext}")
        pattern_upper = os.path.join(images_dir, f"*{ext.upper()}")
        matched_lower = glob.glob(pattern_lower)
        matched_upper = glob.glob(pattern_upper)
        if verbose:
            print(f"Searching for pattern: {pattern_lower} -> Found {len(matched_lower)} files.")
            print(f"Searching for pattern: {pattern_upper} -> Found {len(matched_upper)} files.")
        image_files.update(matched_lower)
        image_files.update(matched_upper)
    image_files = sorted(image_files)
    return list(image_files)

def filter_hidden_files(file_list, verbose=False):
    """
    Remove hidden or system files from the list.
    """
    filtered = []
    for file in file_list:
        basename = os.path.basename(file)
        if not basename.startswith('.'):
            filtered.append(file)
        else:
            if verbose:
                print(f"Excluding hidden/system file: {file}")
    return filtered

def main():
    args = parse_args()

    images_dir = args.images_dir
    labels_dir = args.labels_dir
    output_dir = args.output_dir
    class_names = args.class_names
    image_extensions = normalize_extensions(args.image_extensions)
    verbose = args.verbose

    if not os.path.isdir(images_dir):
        print(f"Error: Images directory '{images_dir}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(labels_dir):
        print(f"Error: Labels directory '{labels_dir}' does not exist.")
        sys.exit(1)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"Output directory '{output_dir}' is ready.")

    image_files = get_image_files(images_dir, image_extensions, verbose=verbose)
    image_files = filter_hidden_files(image_files, verbose=verbose)

    if not image_files:
        print(f"No images found in '{images_dir}' with extensions {image_extensions}.")
        sys.exit(1)

    total_images = len(image_files)
    print(f"Found {total_images} image(s) in '{images_dir}'.")

    current_index = 0

    while True:
        image_path = image_files[current_index]
        image_name = os.path.basename(image_path)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image '{image_path}'. Skipping.")
            current_index += 1
            if current_index >= total_images:
                print("Reached the end of the image list.")
                break
            continue

        # Load labels
        labels = load_labels(label_path, verbose=verbose)

        # Visualize
        annotated_image = visualize(image.copy(), labels, class_names)

        # If output_dir is specified, save the annotated image
        if output_dir:
            output_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_path, annotated_image)
            if verbose:
                print(f"Saved annotated image to '{output_path}'.")
            # Move to next image automatically
            current_index += 1
            if current_index >= total_images:
                print("All images have been processed and saved.")
                break
        else:
            # Display the image and wait for user input
            window_title = f"Image {current_index + 1}/{total_images}: {image_name}"
            cv2.imshow(window_title, annotated_image)
            print(f"Displaying image {current_index + 1}/{total_images}: {image_name}")
            print("Press 'n' for next, 'p' for previous, 'q' to quit.")

            key = cv2.waitKey(0) & 0xFF

            # Close the current window
            cv2.destroyAllWindows()

            if key == ord('n'):
                current_index += 1
                if current_index >= total_images:
                    print("Reached the end of the image list.")
                    break
            elif key == ord('p'):
                current_index -= 1
                if current_index < 0:
                    print("Already at the first image.")
                    current_index = 0
            elif key == ord('q'):
                print("Exiting.")
                break
            else:
                print("Unrecognized key. Use 'n', 'p', or 'q'.")

    # If saving images, notify completion
    if output_dir:
        print(f"All annotated images have been saved to '{output_dir}'.")

if __name__ == "__main__":
    main()