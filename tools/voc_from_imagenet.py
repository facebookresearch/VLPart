# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import json
import random

import imagesize


# imagenet_id starts from 1
voc_categories = [
  {'id': 1, 'name': 'aeroplane', 'imagenet_id': [[405, 405]]},
  {'id': 2, 'name': 'bicycle', 'imagenet_id': [[445, 445], [672, 672]]},
  # {'id': 3, 'name': 'bird', 'imagenet_id': [[10, 25]]}, # too many, no need in ablation study
  {'id': 3, 'name': 'bird', 'imagenet_id': [[10, 14]]},
  {'id': 4, 'name': 'boat', 'imagenet_id': [[485, 485]]},
  {'id': 5, 'name': 'bottle', 'imagenet_id': [[441, 441]]},
  {'id': 6, 'name': 'bus', 'imagenet_id': [[780, 780]]},
  {'id': 7, 'name': 'car', 'imagenet_id': [[408, 408], [610, 610], [628, 628], [818, 818]]},
  {'id': 8, 'name': 'cat', 'imagenet_id': [[282, 286]]},
  {'id': 9, 'name': 'chair', 'imagenet_id': [[424, 424]]},
  {'id': 10, 'name': 'cow', 'imagenet_id': [[346, 348]]},
  {'id': 11, 'name': 'dining table', 'imagenet_id': [[533, 533]]},
  # {'id': 12, 'name': 'dog', 'imagenet_id': [[152, 277]]}, # too many, no need in ablation study
  {'id': 12, 'name': 'dog', 'imagenet_id': [[152, 156]]},
  {'id': 13, 'name': 'horse', 'imagenet_id': []},
  {'id': 14, 'name': 'motorbike', 'imagenet_id': [[671, 671]]},
  {'id': 15, 'name': 'person', 'imagenet_id': []},
  {'id': 16, 'name': 'potted plant', 'imagenet_id': []},
  {'id': 17, 'name': 'sheep', 'imagenet_id': [[349, 354]]},
  {'id': 18, 'name': 'sofa', 'imagenet_id': []},
  {'id': 19, 'name': 'train', 'imagenet_id': [[467, 467]]},
  {'id': 20, 'name': 'tv monitor', 'imagenet_id': [[762, 762]]},
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--superclass_names_path', default='datasets/metadata/imagenet1k_superclass_names.txt')
    parser.add_argument('--label_path', default='datasets/metadata/imagenet1k_label.txt')
    parser.add_argument('--imagenet_path', default='datasets/imagenet/train')
    parser.add_argument("--out_path", default='datasets/imagenet/imagenet_voc_image_info.json')
    args = parser.parse_args()

    superclass_names = []
    f = open(args.superclass_names_path)
    for i, line in enumerate(f):
        superclass_names.append(line[:-1])

    wnids = []
    f = open(args.label_path)
    for i, line in enumerate(f):
        wnids.append(line.split(' ')[0])
    wnids = sorted(wnids)

    image_count = 0
    images = []
    anns = []
    for voc_cat in voc_categories:
        imagenet_id_pairs = voc_cat['imagenet_id']
        imagenet_id_set = []
        for (id_start, id_end) in imagenet_id_pairs:
            for imagenet_id in range(id_start - 1, id_end + 1 - 1):  # imagenet_id starts from 1
                imagenet_id_set.append(imagenet_id)
        for imagenet_id in imagenet_id_set:
            class_path = os.path.join(args.imagenet_path, wnids[imagenet_id])
            files = sorted(os.listdir(class_path))
            print(superclass_names[imagenet_id], "--",
                  voc_cat['name'], wnids[imagenet_id],
                  len(files))
            for file_name in files:
                image_count = image_count + 1
                width, height = imagesize.get('{}/{}'.format(class_path, file_name))
                if width > 1000 or height > 1000:
                    continue
                image = {
                    'id': image_count,
                    'file_name': os.path.join(wnids[imagenet_id], file_name),
                    'pos_category_ids': [voc_cat['id']],
                    'width': width,
                    'height': height,
                }
                images.append(image)

    data = {'categories': voc_categories, 'images': images, 'annotations': anns}
    for k, v in data.items():
        print(k, len(v))
    print('Saving to ', args.out_path)
    json.dump(data, open(args.out_path, 'w'))
