# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt


def polygons_to_bitmask(polygons, height, width):
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_file', default='datasets/imagenet/imagenet_golden_addition_parsed.json')
    parser.add_argument('--image_folder', default='datasets/imagenet/train')
    # parser.add_argument('--ann_file', default='datasets/imagenet/imagenet_golden_paco_parsed.json')
    # parser.add_argument('--image_folder', default='datasets/imagenet/train')
    # parser.add_argument('--ann_file', default='datasets/imagenet/imagenet_golden_partimagenet_parsed.json')
    # parser.add_argument('--image_folder', default='datasets/imagenet/train')
    # parser.add_argument('--ann_file', default='datasets/imagenet/imagenet_golden_pascal_parsed.json')
    # parser.add_argument('--image_folder', default='datasets/imagenet/train')
    # parser.add_argument('--ann_file', default='datasets/imagenet/imagenet_voc_image_parsed.json')
    # parser.add_argument('--image_folder', default='datasets/imagenet/train')
    # parser.add_argument('--ann_file', default='datasets/partimagenet/partimagenet_parsed.json')
    # parser.add_argument('--image_folder', default='datasets/partimagenet/train')
    parser.add_argument('--interval', default=100)
    parser.add_argument('--output_path', default='output_ann_parsed')
    args = parser.parse_args()

    coco = COCO(args.ann_file)
    for img_idx, img_key in enumerate(coco.imgs):
        if img_idx % args.interval != 0:
            continue
        img = coco.loadImgs([img_key])[0]
        anns = coco.imgToAnns[img['id']]

        img_path = os.path.join(args.image_folder, img['file_name'])
        print(img_path)
        image_show = cv2.imread(img_path)
        height, width, _ = image_show.shape
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(anns)))[:,:3]
        color_maps = [color * 255 for color in colors]

        for instance_id, ann in enumerate(anns):
            instance_color = color_maps[instance_id % len(color_maps)]
            bbox = ann['bbox']
            image_show = cv2.rectangle(
                image_show,
                (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                instance_color.tolist(), 2)
            category_id = ann['category_id']
            image_show = cv2.putText(image_show, "{}".format(coco.loadCats(category_id)[0]['name']),
                                     (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_COMPLEX,
                                     0.5, instance_color.tolist(), 1, cv2.LINE_AA)

            if "segmentation" in ann:
                segm = ann["segmentation"]
                if isinstance(segm, list): # polygon
                    mask = polygons_to_bitmask(segm, height, width)
                elif isinstance(segm, dict): # COCO RLE
                    mask = mask_util.decode(segm)
                else:
                    raise NotImplementedError
                mask_ = mask.astype(np.bool_)
                for color_channel in range(3):
                    image_show[mask_, color_channel] = \
                        image_show[mask_, color_channel] * 0.3 + \
                        instance_color[color_channel] * 0.7

        os.makedirs(args.output_path, exist_ok=True)
        cv2.imwrite(os.path.join(args.output_path, img["file_name"].split('/')[-1]), image_show)
