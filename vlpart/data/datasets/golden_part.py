# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.coco import load_coco_json


ADDITIONAL_PART_CATEGORIES = [
    {"id": 1, "name": "elephant:head"},
    {"id": 2, "name": "elephant:torso"},
    {"id": 3, "name": "elephant:leg"},
    {"id": 4, "name": "elephant:paw"},
    {"id": 5, "name": "elephant:tail"},

    {"id": 6, "name": "koala:head"},
    {"id": 7, "name": "koala:torso"},
    {"id": 8, "name": "koala:leg"},
    {"id": 9, "name": "koala:foot"},
    {"id": 10, "name": "koala:tail"},

    {"id": 11, "name": "fox:head"},
    {"id": 12, "name": "fox:torso"},
    {"id": 13, "name": "fox:leg"},
    {"id": 14, "name": "fox:paw"},
    {"id": 15, "name": "fox:tail"},

    {"id": 16, "name": "leopard:head"},
    {"id": 17, "name": "leopard:torso"},
    {"id": 18, "name": "leopard:leg"},
    {"id": 19, "name": "leopard:paw"},
    {"id": 20, "name": "leopard:tail"},

    {"id": 21, "name": "tiger:head"},
    {"id": 22, "name": "tiger:torso"},
    {"id": 23, "name": "tiger:leg"},
    {"id": 24, "name": "tiger:paw"},
    {"id": 25, "name": "tiger:tail"},

    {"id": 26, "name": "lion:head"},
    {"id": 27, "name": "lion:torso"},
    {"id": 28, "name": "lion:leg"},
    {"id": 29, "name": "lion:paw"},
    {"id": 30, "name": "lion:tail"},

    {"id": 31, "name": "bear:head"},
    {"id": 32, "name": "bear:torso"},
    {"id": 33, "name": "bear:leg"},
    {"id": 34, "name": "bear:paw"},
    {"id": 35, "name": "bear:tail"},

    {"id": 36, "name": "zebra:head"},
    {"id": 37, "name": "zebra:torso"},
    {"id": 38, "name": "zebra:leg"},
    {"id": 39, "name": "zebra:foot"},
    {"id": 40, "name": "zebra:tail"},

    {"id": 41, "name": "pig:head"},
    {"id": 42, "name": "pig:torso"},
    {"id": 43, "name": "pig:leg"},
    {"id": 44, "name": "pig:paw"},
    {"id": 45, "name": "pig:tail"},

    {"id": 46, "name": "monkey:head"},
    {"id": 47, "name": "monkey:torso"},
    {"id": 48, "name": "monkey:leg"},
    {"id": 49, "name": "monkey:paw"},
    {"id": 50, "name": "monkey:tail"},

    {"id": 51, "name": "panda:head"},
    {"id": 52, "name": "panda:torso"},
    {"id": 53, "name": "panda:leg"},
    {"id": 54, "name": "panda:paw"},
    {"id": 55, "name": "panda:tail"},
]


def _get_metadata():
    id_to_name = {x['id']: x['name'] for x in ADDITIONAL_PART_CATEGORIES}
    thing_dataset_id_to_contiguous_id = {
        x: i for i, x in enumerate(sorted(id_to_name))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


def register_additional_part_instances(name, metadata, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(
        json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="pascal_part", **metadata
    )

_ADDITIONAL_PART = {
    "imagenet_golden_addition_parsed": ("imagenet/train", "imagenet/imagenet_golden_addition_parsed.json"),
}

def register_all_additional_part(root):
    for key, (image_root, json_file) in _ADDITIONAL_PART.items():
        register_additional_part_instances(
            key,
            _get_metadata(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_additional_part(_root)
