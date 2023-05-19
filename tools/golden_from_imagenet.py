# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import json
import random
import sys
sys.path.append('.')
from vlpart.data.datasets.golden_categories import (
    PASCAL_GOLDEN_CATEGORIES,
    PARTIMAGENET_GOLDEN_CATEGORIES,
    PACO_GOLDEN_CATEGORIES,
    GOLDEN_CATEGORIES,
    ADDITIONAL_CATEGORIES,
)
import imagesize


CATEGORY_SET_DICT = {
    "all": GOLDEN_CATEGORIES + ADDITIONAL_CATEGORIES,
    "pascal": PASCAL_GOLDEN_CATEGORIES,
    "partimagenet": PARTIMAGENET_GOLDEN_CATEGORIES,
    "paco": PACO_GOLDEN_CATEGORIES,
    "golden": GOLDEN_CATEGORIES,
    "addition": ADDITIONAL_CATEGORIES,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', default='datasets/metadata/imagenet1k_label.txt')
    parser.add_argument('--imagenet_path', default='datasets/imagenet/train')
    parser.add_argument('--category_set', default='all')
    parser.add_argument('--image_per_folder', default=500)
    parser.add_argument("--out_path", default='datasets/imagenet/imagenet_golden_image_info.json')
    args = parser.parse_args()

    wnids = []
    f = open(args.label_path)
    for i, line in enumerate(f):
        wnids.append(line.split(' ')[0])
    wnids = sorted(wnids)

    image_count = 0
    images = []
    anns = []
    golden_categories = CATEGORY_SET_DICT[args.category_set]

    for golden_cat in golden_categories:
        print(golden_cat['name'])
        imagenet_id_pairs = golden_cat['imagenet_id']
        imagenet_id_set = []
        for (id_start, id_end) in imagenet_id_pairs:
            for imagenet_id in range(id_start - 1, id_end + 1 - 1):  # imagenet_id starts from 1
                imagenet_id_set.append(imagenet_id)
        for imagenet_id in imagenet_id_set:
            class_path = os.path.join(args.imagenet_path, wnids[imagenet_id])
            files = sorted(os.listdir(class_path))
            for file_name in files[:args.image_per_folder]:
                image_count = image_count + 1
                width, height = imagesize.get('{}/{}'.format(class_path, file_name))
                if width > 1000 or height > 1000:
                    continue
                image = {
                    'id': image_count,
                    'file_name': os.path.join(wnids[imagenet_id], file_name),
                    'pos_category_ids': [golden_cat['id']],
                    'width': width,
                    'height': height,
                }
                images.append(image)

    data = {'categories': golden_categories, 'images': images, 'annotations': anns}
    for k, v in data.items():
        print(k, len(v))
    print('Saving to ', args.out_path)
    json.dump(data, open(args.out_path, 'w'))
