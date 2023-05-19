# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

from detectron2.data.datasets.register_coco import register_coco_instances

PASCAL_PART_CATEGORIES = [
    {"id": 0, "name": "aeroplane:body"},
    {"id": 1, "name": "aeroplane:wing"},
    {"id": 2, "name": "aeroplane:tail"},
    {"id": 3, "name": "aeroplane:wheel"},
    {"id": 4, "name": "bicycle:wheel"},
    {"id": 5, "name": "bicycle:handlebar"},
    {"id": 6, "name": "bicycle:saddle"},
    {"id": 7, "name": "bird:beak"},
    {"id": 8, "name": "bird:head"},
    {"id": 9, "name": "bird:eye"},
    {"id": 10, "name": "bird:leg"},
    {"id": 11, "name": "bird:foot"},
    {"id": 12, "name": "bird:wing"},
    {"id": 13, "name": "bird:neck"},
    {"id": 14, "name": "bird:tail"},
    {"id": 15, "name": "bird:torso"},
    {"id": 16, "name": "bottle:body"},
    {"id": 17, "name": "bottle:cap"},
    {"id": 18, "name": "bus:license plate", "abbr": "bus:liplate"},
    {"id": 19, "name": "bus:headlight"},
    {"id": 20, "name": "bus:door"},
    {"id": 21, "name": "bus:mirror"},
    {"id": 22, "name": "bus:window"},
    {"id": 23, "name": "bus:wheel"},
    {"id": 24, "name": "car:license plate", "abbr": "car:liplate"},
    {"id": 25, "name": "car:headlight"},
    {"id": 26, "name": "car:door"},
    {"id": 27, "name": "car:mirror"},
    {"id": 28, "name": "car:window"},
    {"id": 29, "name": "car:wheel"},
    {"id": 30, "name": "cat:head"},
    {"id": 31, "name": "cat:leg"},
    {"id": 32, "name": "cat:ear"},
    {"id": 33, "name": "cat:eye"},
    {"id": 34, "name": "cat:paw", "abbr": "cat:pa"},
    {"id": 35, "name": "cat:neck"},
    {"id": 36, "name": "cat:nose"},
    {"id": 37, "name": "cat:tail"},
    {"id": 38, "name": "cat:torso"},
    {"id": 39, "name": "cow:head"},
    {"id": 40, "name": "cow:leg"},
    {"id": 41, "name": "cow:ear"},
    {"id": 42, "name": "cow:eye"},
    {"id": 43, "name": "cow:neck"},
    {"id": 44, "name": "cow:horn"},
    {"id": 45, "name": "cow:muzzle"},
    {"id": 46, "name": "cow:tail"},
    {"id": 47, "name": "cow:torso"},
    {"id": 48, "name": "dog:head"},
    {"id": 49, "name": "dog:leg"},
    {"id": 50, "name": "dog:ear"},
    {"id": 51, "name": "dog:eye"},
    {"id": 52, "name": "dog:paw", "abbr": "dog:pa"},
    {"id": 53, "name": "dog:neck"},
    {"id": 54, "name": "dog:nose"},
    {"id": 55, "name": "dog:muzzle"},
    {"id": 56, "name": "dog:tail"},
    {"id": 57, "name": "dog:torso"},
    {"id": 58, "name": "horse:head"},
    {"id": 59, "name": "horse:leg"},
    {"id": 60, "name": "horse:ear"},
    {"id": 61, "name": "horse:eye"},
    {"id": 62, "name": "horse:neck"},
    {"id": 63, "name": "horse:muzzle"},
    {"id": 64, "name": "horse:tail"},
    {"id": 65, "name": "horse:torso"},
    {"id": 66, "name": "motorbike:wheel"},
    {"id": 67, "name": "motorbike:handlebar"},
    {"id": 68, "name": "motorbike:headlight"},
    {"id": 69, "name": "motorbike:saddle"},
    {"id": 70, "name": "pottedplant:plant"},
    {"id": 71, "name": "pottedplant:pot"},
    {"id": 72, "name": "sheep:head"},
    {"id": 73, "name": "sheep:leg"},
    {"id": 74, "name": "sheep:ear"},
    {"id": 75, "name": "sheep:eye"},
    {"id": 76, "name": "sheep:neck"},
    {"id": 77, "name": "sheep:horn"},
    {"id": 78, "name": "sheep:muzzle"},
    {"id": 79, "name": "sheep:tail"},
    {"id": 80, "name": "sheep:torso"},
]

PASCAL_PART_DOG_CATEGORIES = [
    {"id": 0, "name": "dog:head"},
    {"id": 1, "name": "dog:leg"},
    {"id": 2, "name": "dog:ear"},
    {"id": 3, "name": "dog:eye"},
    {"id": 4, "name": "dog:paw", "abbr": "dog:pa"},
    {"id": 5, "name": "dog:neck"},
    {"id": 6, "name": "dog:nose"},
    {"id": 7, "name": "dog:muzzle"},
    {"id": 8, "name": "dog:tail"},
    {"id": 9, "name": "dog:torso"},
]


def _get_partimagenet_metadata(only_dog=False):
    if only_dog:
        id_to_name = {x['id']: x['name'] for x in PASCAL_PART_DOG_CATEGORIES}
    else:
        id_to_name = {x['id']: x['name'] for x in PASCAL_PART_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


_PASCAL_PART = {
    "pascal_part_train": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train.json"),
    "pascal_part_val": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/val.json"),
    "pascal_part_dog_train": ("pascal_part/VOCdevkit/VOC2010/JPEGImages", "pascal_part/train_dog.json"),
}


def register_all_pascal_part(root):
    for key, (image_root, json_file) in _PASCAL_PART.items():
        register_coco_instances(
            key,
            _get_partimagenet_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
