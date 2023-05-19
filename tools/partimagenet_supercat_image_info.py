# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import torch
import numpy as np
import json
from collections import defaultdict


PARTIMAGENET_CATEGORIES = [
    {"id": 0, "name": "Quadruped Head"},
    {"id": 1, "name": "Quadruped Body"},
    {"id": 2, "name": "Quadruped Foot"},
    {"id": 3, "name": "Quadruped Tail"},
    {"id": 4, "name": "Biped Head"},
    {"id": 5, "name": "Biped Body"},
    {"id": 6, "name": "Biped Hand"},
    {"id": 7, "name": "Biped Foot"},
    {"id": 8, "name": "Biped Tail"},
    {"id": 9, "name": "Fish Head"},
    {"id": 10, "name": "Fish Body"},
    {"id": 11, "name": "Fish Fin"},
    {"id": 12, "name": "Fish Tail"},
    {"id": 13, "name": "Bird Head"},
    {"id": 14, "name": "Bird Body"},
    {"id": 15, "name": "Bird Wing"},
    {"id": 16, "name": "Bird Foot"},
    {"id": 17, "name": "Bird Tail"},
    {"id": 18, "name": "Snake Head"},
    {"id": 19, "name": "Snake Body"},
    {"id": 20, "name": "Reptile Head"},
    {"id": 21, "name": "Reptile Body"},
    {"id": 22, "name": "Reptile Foot"},
    {"id": 23, "name": "Reptile Tail"},
    {"id": 24, "name": "Car Body"},
    {"id": 25, "name": "Car Tier"},
    {"id": 26, "name": "Car Side Mirror"},
    {"id": 27, "name": "Bicycle Body"},
    {"id": 28, "name": "Bicycle Head"},
    {"id": 29, "name": "Bicycle Seat"},
    {"id": 30, "name": "Bicycle Tier"},
    {"id": 31, "name": "Boat Body"},
    {"id": 32, "name": "Boat Sail"},
    {"id": 33, "name": "Aeroplane Head"},
    {"id": 34, "name": "Aeroplane Body"},
    {"id": 35, "name": "Aeroplane Engine"},
    {"id": 36, "name": "Aeroplane Wing"},
    {"id": 37, "name": "Aeroplane Tail"},
    {"id": 38, "name": "Bottle Mouth"},
    {"id": 39, "name": "Bottle Body"},
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cat_json_path', default='datasets/partimagenet/train_format.json')
    parser.add_argument('--supercat_json_path', default='datasets/partimagenet/partimagenet_supercat_image_info.json')
    args = parser.parse_args()

    super_cat_names = []
    for x in PARTIMAGENET_CATEGORIES:
        if x['name'].split(' ')[0] not in super_cat_names:
            super_cat_names.append(x['name'].split(' ')[0])

    super_cat_name2id = {}
    super_cat_id = 0
    for super_cat_name in super_cat_names:
        super_cat_name2id[super_cat_name] = super_cat_id
        super_cat_id += 1

    cat_id2supercat_id = {}
    for cat in PARTIMAGENET_CATEGORIES:
        cat_id2supercat_id[cat['id']] = super_cat_name2id[cat['name'].split(' ')[0]]

    print(cat_id2supercat_id)

    new_cat = []
    for super_cat_name in super_cat_name2id:
        super_cat = {
            'id': super_cat_name2id[super_cat_name],
            'name': super_cat_name,
        }
        new_cat.append(super_cat)
    print(new_cat)

    print('Loading cat_json')
    old_json = json.load(open(args.cat_json_path, 'r'))

    imageid2ann = defaultdict(list)
    for ann in old_json['annotations']:
        imageid2ann[ann['image_id']].append(ann['category_id'])

    imageid2image_info = {}
    for image_info in old_json['images']:
        imageid2image_info[image_info['id']] = image_info

    new_images = []
    for image_id in imageid2ann:
        cat_id = imageid2ann[image_id][0]  # only consider first ann
        supercat_id = cat_id2supercat_id[cat_id]
        image_info = imageid2image_info[image_id]
        image_info['pos_category_ids'] = [supercat_id]
        new_images.append(image_info)

    new_json = {
        'images': new_images,
        'annotations': [],
        'categories': new_cat,
    }
    for k, v in new_json.items():
        print(k, len(v))
    print('Writing to', args.supercat_json_path)
    json.dump(new_json, open(args.supercat_json_path, 'w'))
