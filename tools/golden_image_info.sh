#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# script to collect image_info

python tools/golden_from_imagenet.py
python tools/golden_from_imagenet.py --category_set pascal --out_path 'datasets/imagenet/imagenet_pascal_image_info.json'
python tools/golden_from_imagenet.py --category_set partimagenet --out_path 'datasets/imagenet/imagenet_partimagenet_image_info.json'
python tools/golden_from_imagenet.py --category_set paco --out_path 'datasets/imagenet/imagenet_paco_image_info.json'
python tools/golden_from_imagenet.py --category_set addition --out_path 'datasets/imagenet/imagenet_addition_image_info.json'
