# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import os
import xml.etree.ElementTree as ET

CLASS_NAMES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
)

novel_names = (
    "bus", "dog",
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_path', default='datasets/pascal_part/VOCdevkit/VOC2010/Annotations')
    parser.add_argument("--part_path", default='datasets/pascal_part/train.json')
    parser.add_argument("--out_path", default='datasets/pascal_part/train_one.json')
    parser.add_argument('--only_base', action='store_true')
    args = parser.parse_args()

    filename2num = {}
    files = sorted(os.listdir(args.voc_path))
    for filename in files:
        anno_file = '{}/{}'.format(args.voc_path, filename)
        tree = ET.parse(open(anno_file))
        num_obj = 0
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if args.only_base and cls in novel_names:
                num_obj = 0
                break
            num_obj += 1
        image_filename = filename.split('.')[0]
        filename2num[image_filename] = num_obj

    part_ann = json.load(open(args.part_path))
    print('Before:')
    for k, v in part_ann.items():
        print(k, len(v))

    new_images = []
    new_annos = []
    valid_image_id = []
    for image_info in part_ann['images']:
        image_filename = image_info['file_name'].split('.')[0]
        if image_filename in filename2num and filename2num[image_filename] == 1:
            new_images.append(image_info)
            valid_image_id.append(image_info['id'])

    for ann_info in part_ann['annotations']:
        if ann_info['image_id'] in valid_image_id:
            new_annos.append(ann_info)

    data = {'categories': part_ann['categories'], 'images': new_images, 'annotations': new_annos}
    print('After:')
    for k, v in data.items():
        print(k, len(v))
    print('Saving to ', args.out_path)
    json.dump(data, open(args.out_path, 'w'))
