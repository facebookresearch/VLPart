# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

from detectron2.data.datasets.register_coco import register_coco_instances


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


PARTIMAGENET_SUPER_CATEGORIES = [
    {"id": 0, "name": "Quadruped"},
    {"id": 1, "name": "Biped"},
    {"id": 2, "name": "Fish"},
    {"id": 3, "name": "Bird"},
    {"id": 4, "name": "Snake"},
    {"id": 5, "name": "Reptile"},
    {"id": 6, "name": "Car"},
    {"id": 7, "name": "Bicycle"},
    {"id": 8, "name": "Boat"},
    {"id": 9, "name": "Aeroplane"},
    {"id": 10, "name": "Bottle"},
]

def _get_partimagenet_metadata(key):
    if 'supercat' in key:
        id_to_name = {x['id']: x['name'] for x in PARTIMAGENET_SUPER_CATEGORIES}
    else:
        id_to_name = {x['id']: x['name'] for x in PARTIMAGENET_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PARTIMAGENET = {
    "partimagenet_train": ("partimagenet/train/", "partimagenet/train_format.json"),
    "partimagenet_val": ("partimagenet/val/", "partimagenet/val_format.json"),
    "partimagenet_parsed": ("partimagenet/train", "partimagenet/partimagenet_parsed.json"),
    "imagenet_golden_partimagenet_parsed": ("imagenet/train", "imagenet/imagenet_golden_partimagenet_parsed.json"),
    "imagenet_golden_partimagenet_parsed_swinbase": ("imagenet/train", "imagenet/imagenet_golden_partimagenet_parsed_swinbase.json"),

}

def register_all_partimagenet(root):
    for key, (image_root, json_file) in _PARTIMAGENET.items():
        register_coco_instances(
            key,
            _get_partimagenet_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_partimagenet(_root)
