# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis import get_lvis_instances_meta
from .lvis_v1 import custom_load_lvis_json
from .voc import _get_builtin_metadata as get_voc_instances_meta
from .partimagenet import _get_partimagenet_metadata as get_partimagenet_instances_meta
from .golden_categories import _get_builtin_metadata as get_golden_instances_meta


def _get_imagenet_metadata(key):
    if 'lvis_v1' in key:
        return get_lvis_instances_meta('lvis_v1')
    elif 'voc' in key:
        return get_voc_instances_meta()
    elif 'partimagenet' in key:
        return get_partimagenet_instances_meta(key)
    elif 'golden' in key:
        return get_golden_instances_meta()


def custom_register_imagenet_instances(name, metadata, json_file, image_root, random_image=False):
    DatasetCatalog.register(name, lambda: custom_load_lvis_json(
        json_file, image_root, name, random_image))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="imagenet", **metadata
    )


_CUSTOM_SPLITS_IMAGENET = {
    "imagenet_lvis_v1": ("imagenet/ImageNet-LVIS/", "imagenet/annotations/imagenet_lvis_image_info.json", False),
    "imagenet_voc": ("imagenet/train/", "imagenet/imagenet_voc_image_info.json", False),
    "imagenet_voc_random": ("imagenet/train/", "imagenet/imagenet_voc_image_info.json", True),
    "partimagenet_supercat": ("partimagenet/train/", "partimagenet/partimagenet_supercat_image_info.json", False),
    "partimagenet_supercat_random": ("partimagenet/train/", "partimagenet/partimagenet_supercat_image_info.json", True),
    "imagenet_golden": ("imagenet/train/", "imagenet/imagenet_golden_image_info.json", False),
    "imagenet_golden_pascal": ("imagenet/train/", "imagenet/imagenet_pascal_image_info.json", False),
    "imagenet_golden_partimagenet": ("imagenet/train/", "imagenet/imagenet_partimagenet_image_info.json", False),
    "imagenet_golden_paco": ("imagenet/train/", "imagenet/imagenet_paco_image_info.json", False),
    "imagenet_golden_addition": ("imagenet/train/", "imagenet/imagenet_addition_image_info.json", False),
}


def register_all_lvis_imagenet(root):
    for key, (image_root, json_file, random_image) in _CUSTOM_SPLITS_IMAGENET.items():
        custom_register_imagenet_instances(
            key,
            _get_imagenet_metadata(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            random_image,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_lvis_imagenet(_root)
