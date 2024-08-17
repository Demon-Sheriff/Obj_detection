import os
import cv2
import random
import shutil
import argparse

"""
This is a python script to crop person images along with
the annotations and save the results after splitting the dataset
into train and val for training the model.
"""

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8):
    # Create output directories
    train_images_dir = os.path.join(output_dir, 'train/images')
    val_images_dir = os.path.join(output_dir, 'val/images')
    train_labels_dir = os.path.join(output_dir, 'train/labels')
    val_labels_dir = os.path.join(output_dir, 'val/labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Get list of all images
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # Shuffle the images to ensure random split
    random.shuffle(images)

    # Split the data
    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]

    # Copy files to the respective directories
    for image in train_images:
        shutil.copy(os.path.join(image_dir, image), os.path.join(train_images_dir, image))
        label = image.replace('.jpg', '.txt')
        shutil.copy(os.path.join(label_dir, label), os.path.join(train_labels_dir, label))

    for image in val_images:
        shutil.copy(os.path.join(image_dir, image), os.path.join(val_images_dir, image))
        label = image.replace('.jpg', '.txt')
        shutil.copy(os.path.join(label_dir, label), os.path.join(val_labels_dir, label))

    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")


def crop_person_and_adjust_annotations(input_dir, annotations_dir, output_dir, classes_file):
    with open(classes_file, 'r') as f:
        class_names = f.read().splitlines()

    os.makedirs(output_dir, exist_ok=True)
    cropped_annotations_dir = os.path.join(output_dir, 'cropped_annotations')
    os.makedirs(cropped_annotations_dir, exist_ok=True)
    
    for file in os.listdir(annotations_dir):
        if file.endswith('.txt'):
            img_file = file.replace('.txt', '.jpg')  # Assuming images are in jpg format
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            
            with open(os.path.join(annotations_dir, file), 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                cls_id, x, y, w, h = map(float, line.strip().split())
                if class_names[int(cls_id)] == 'person':
                    # Convert YOLO format to OpenCV format
                    x_center, y_center = int(x * img.shape[1]), int(y * img.shape[0])
                    box_w, box_h = int(w * img.shape[1]), int(h * img.shape[0])
                    x1, y1 = int(x_center - box_w / 2), int(y_center - box_h / 2)
                    x2, y2 = int(x_center + box_w / 2), int(y_center + box_h / 2)
                    
                    # Crop the image
                    cropped_img = img[y1:y2, x1:x2]
                    cropped_img_file = f"{file.replace('.txt', '')}_crop_{i}.jpg"
                    cv2.imwrite(os.path.join(output_dir, cropped_img_file), cropped_img)
                    
                    # Adjust annotations for the cropped image
                    cropped_annotations = []
                    for line in lines:
                        cls_id, x, y, w, h = map(float, line.strip().split())
                        if class_names[int(cls_id)] != 'person':  # Exclude person class
                            # Convert YOLO coordinates to absolute coordinates in the cropped image
                            abs_x_center = (x * img.shape[1]) - x1
                            abs_y_center = (y * img.shape[0]) - y1
                            abs_x_center /= (x2 - x1)
                            abs_y_center /= (y2 - y1)
                            w /= (x2 - x1) / img.shape[1]
                            h /= (y2 - y1) / img.shape[0]

                            # Only save the annotation if the bounding box is still within the cropped image
                            if 0 <= abs_x_center <= 1 and 0 <= abs_y_center <= 1:
                                cropped_annotations.append(f"{cls_id} {abs_x_center} {abs_y_center} {w} {h}\n")
                    
                    # Save the cropped annotations
                    if cropped_annotations:
                        with open(os.path.join(cropped_annotations_dir, f"{file.replace('.txt', '')}_crop_{i}.txt"), 'w') as f_out:
                            f_out.writelines(cropped_annotations)

    print(f"Processed images and annotations saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop person images and adjust annotations, then split into train/val sets.")
    
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing full images.")
    parser.add_argument("--annotations_dir", type=str, required=True, help="Directory containing full annotations in YOLO format.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for cropped images and annotations.")
    parser.add_argument("--classes_file", type=str, required=True, help="Path to the file containing class names.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of images to use for training (default: 0.8).")

    args = parser.parse_args()

    # Crop and adjust annotations
    crop_person_and_adjust_annotations(
        input_dir=args.input_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        classes_file=args.classes_file
    )

    # Split dataset into training and validation sets
    split_dataset(
        image_dir=os.path.join(args.output_dir, 'images'),
        label_dir=os.path.join(args.output_dir, 'cropped_annotations'),
        output_dir=os.path.join(args.output_dir, 'split'),
        train_ratio=args.train_ratio
    )
