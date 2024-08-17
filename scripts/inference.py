import cv2
import os
import argparse
from ultralytics import YOLO

def perform_inference(input_dir, output_dir, person_model, ppe_model):
    # Load the models
    person_detector = YOLO(person_model)
    ppe_detector = YOLO(ppe_model)
    
    # Create output directories for train and val sets
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

    # Iterate through train and val directories
    for split in ['train', 'val']:
        split_input_dir = os.path.join(input_dir, split, 'images')
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)

        for img_file in os.listdir(split_input_dir):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                img_path = os.path.join(split_input_dir, img_file)
                img = cv2.imread(img_path)
                
                # Perform person detection
                results_person = person_detector(img)
                for result in results_person:
                    if result['class'] == 'person':
                        x1, y1, x2, y2 = result['bbox']
                        cropped_img = img[y1:y2, x1:x2]
                        
                        # Perform PPE detection
                        results_ppe = ppe_detector(cropped_img)
                        for ppe_result in results_ppe:
                            ppe_x1, ppe_y1, ppe_x2, ppe_y2 = ppe_result['bbox']
                            
                            # Adjust coordinates to the full image
                            full_x1, full_y1 = x1 + ppe_x1, y1 + ppe_y1
                            full_x2, full_y2 = x1 + ppe_x2, y1 + ppe_y2
                            
                            # Draw bounding boxes on the original image
                            cv2.rectangle(img, (full_x1, full_y1), (full_x2, full_y2), (255, 0, 0), 2)
                            cv2.putText(img, f"{ppe_result['class']} {ppe_result['confidence']:.2f}",
                                        (full_x1, full_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                # Save the image with predictions
                output_img_path = os.path.join(split_output_dir, img_file)
                cv2.imwrite(output_img_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on images using YOLO models.")
    parser.add_argument('input_dir', type=str, help="Directory containing train and val images.")
    parser.add_argument('output_dir', type=str, help="Directory to save output images with predictions.")
    parser.add_argument('person_det_model', type=str, help="Path to the person detection model weights.")
    parser.add_argument('ppe_detection_model', type=str, help="Path to the PPE detection model weights.")
    
    args = parser.parse_args()
    perform_inference(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)
