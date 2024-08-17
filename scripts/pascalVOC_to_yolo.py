import os
import xml.etree.ElementTree as ET
import argparse

def convert_pascal_voc_to_yolo(voc_dir, output_dir, classes_file):
    with open(classes_file, 'r') as f:
        class_names = f.read().splitlines()

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(voc_dir):
        if file.endswith('.xml'):
            tree = ET.parse(os.path.join(voc_dir, file))
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            with open(os.path.join(output_dir, file.replace('.xml', '.txt')), 'w') as f_out:
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in class_names:
                        continue
                    cls_id = class_names.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert_bbox((width, height), b)
                    f_out.write(f"{cls_id} " + " ".join([f"{a:.6f}" for a in bb]) + '\n')

def convert_bbox(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC annotations to YOLOv8 format.")
    parser.add_argument('voc_dir', type=str, help="Directory containing PascalVOC XML files.")
    parser.add_argument('output_dir', type=str, help="Output directory for YOLOv8 annotations.")
    parser.add_argument('classes_file', type=str, help="Path to the classes.txt file.")

    args = parser.parse_args()
    convert_pascal_voc_to_yolo(args.voc_dir, args.output_dir, args.classes_file)
