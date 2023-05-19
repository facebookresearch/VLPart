# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import json
import scipy.io as sio
import numpy as np
from collections import defaultdict
import pycocotools.mask as mask_util

PASCAL_PART_CATEGORIES = [
    {"id": 1, "name": "aeroplane:body"},
    {"id": 2, "name": "aeroplane:wing"},
    {"id": 3, "name": "aeroplane:tail"},
    {"id": 4, "name": "aeroplane:wheel"},
    {"id": 5, "name": "bicycle:wheel"},
    {"id": 6, "name": "bicycle:handlebar"},
    {"id": 7, "name": "bicycle:saddle"},
    {"id": 8, "name": "bird:beak"},
    {"id": 9, "name": "bird:head"},
    {"id": 10, "name": "bird:eye"},
    {"id": 11, "name": "bird:leg"},
    {"id": 12, "name": "bird:foot"},
    {"id": 13, "name": "bird:wing"},
    {"id": 14, "name": "bird:neck"},
    {"id": 15, "name": "bird:tail"},
    {"id": 16, "name": "bird:torso"},
    {"id": 17, "name": "bottle:body"},
    {"id": 18, "name": "bottle:cap"},
    {"id": 19, "name": "bus:license plate", "abbr": "bus:liplate"},
    {"id": 20, "name": "bus:headlight"},
    {"id": 21, "name": "bus:door"},
    {"id": 22, "name": "bus:mirror"},
    {"id": 23, "name": "bus:window"},
    {"id": 24, "name": "bus:wheel"},
    {"id": 25, "name": "car:license plate", "abbr": "car:liplate"},
    {"id": 26, "name": "car:headlight"},
    {"id": 27, "name": "car:door"},
    {"id": 28, "name": "car:mirror"},
    {"id": 29, "name": "car:window"},
    {"id": 30, "name": "car:wheel"},
    {"id": 31, "name": "cat:head"},
    {"id": 32, "name": "cat:leg"},
    {"id": 33, "name": "cat:ear"},
    {"id": 34, "name": "cat:eye"},
    {"id": 35, "name": "cat:paw", "abbr": "cat:pa"},
    {"id": 36, "name": "cat:neck"},
    {"id": 37, "name": "cat:nose"},
    {"id": 38, "name": "cat:tail"},
    {"id": 39, "name": "cat:torso"},
    {"id": 40, "name": "cow:head"},
    {"id": 41, "name": "cow:leg"},
    {"id": 42, "name": "cow:ear"},
    {"id": 43, "name": "cow:eye"},
    {"id": 44, "name": "cow:neck"},
    {"id": 45, "name": "cow:horn"},
    {"id": 46, "name": "cow:muzzle"},
    {"id": 47, "name": "cow:tail"},
    {"id": 48, "name": "cow:torso"},
    {"id": 49, "name": "dog:head"},
    {"id": 50, "name": "dog:leg"},
    {"id": 51, "name": "dog:ear"},
    {"id": 52, "name": "dog:eye"},
    {"id": 53, "name": "dog:paw", "abbr": "dog:pa"},
    {"id": 54, "name": "dog:neck"},
    {"id": 55, "name": "dog:nose"},
    {"id": 56, "name": "dog:muzzle"},
    {"id": 57, "name": "dog:tail"},
    {"id": 58, "name": "dog:torso"},
    {"id": 59, "name": "horse:head"},
    {"id": 60, "name": "horse:leg"},
    {"id": 61, "name": "horse:ear"},
    {"id": 62, "name": "horse:eye"},
    {"id": 63, "name": "horse:neck"},
    {"id": 64, "name": "horse:muzzle"},
    {"id": 65, "name": "horse:tail"},
    {"id": 66, "name": "horse:torso"},
    {"id": 67, "name": "motorbike:wheel"},
    {"id": 68, "name": "motorbike:handlebar"},
    {"id": 69, "name": "motorbike:headlight"},
    {"id": 70, "name": "motorbike:saddle"},
    {"id": 71, "name": "person:hair"},
    {"id": 72, "name": "person:head"},
    {"id": 73, "name": "person:ear"},
    {"id": 74, "name": "person:eye"},
    {"id": 75, "name": "person:nose"},
    {"id": 76, "name": "person:neck"},
    {"id": 77, "name": "person:mouth"},
    {"id": 78, "name": "person:arm"},
    {"id": 79, "name": "person:hand"},
    {"id": 80, "name": "person:leg"},
    {"id": 81, "name": "person:foot"},
    {"id": 82, "name": "person:torso"},
    {"id": 83, "name": "potted plant:plant"},
    {"id": 84, "name": "potted plant:pot"},
    {"id": 85, "name": "sheep:head"},
    {"id": 86, "name": "sheep:leg"},
    {"id": 87, "name": "sheep:ear"},
    {"id": 88, "name": "sheep:eye"},
    {"id": 89, "name": "sheep:neck"},
    {"id": 90, "name": "sheep:horn"},
    {"id": 91, "name": "sheep:muzzle"},
    {"id": 92, "name": "sheep:tail"},
    {"id": 93, "name": "sheep:torso"},
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


def mask2json(part_mask):
    rle = mask_util.encode(np.array(part_mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")

    y, x = np.where(part_mask > 0)
    box = float(np.min(x)), float(np.min(y)), float(np.max(x)), float(np.max(y))
    return rle, [box[0], box[1], box[2]-box[0], box[3]-box[1]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_path', default='DATASET/pascal_part/VOCdevkit/VOC2010/ImageSets/Main')
    parser.add_argument('--split', default='val.txt')
    parser.add_argument('--ann_path', default='DATASET/pascal_part/Annotations_Part')
    parser.add_argument('--ann_out', default='DATASET/pascal_part/val_dog_start1.json')
    parser.add_argument('--only_dog', default=True, action='store_true')
    args = parser.parse_args()

    PART_CATEGORIES = PASCAL_PART_DOG_CATEGORIES if args.only_dog else PASCAL_PART_CATEGORIES
    for cat in PART_CATEGORIES:
        cat['id'] = cat['id'] + 1

    objname2partidset = defaultdict(list)
    partid2partname = dict()
    for objpart in PART_CATEGORIES:
        objname2partidset[objpart['name'].split(":")[0]].append(objpart['id'])
        partid2partname[objpart['id']] = objpart['name'] if 'abbr' not in objpart else objpart['abbr']
    part_num = {objpart['id']: 0 for objpart in PART_CATEGORIES}

    with open(os.path.join(args.split_path, args.split)) as f:
        img_pathset = np.loadtxt(f, dtype=str)

    ann_json = {}
    ann_json['categories'] = PART_CATEGORIES
    imgs = []
    anns = []
    json_ann_id = 0
    for img_path in img_pathset:
        file = sio.loadmat(os.path.join(args.ann_path, img_path))
        img_anno = file['anno'][0][0]
        imname = str(img_anno[0][0])
        print(imname)
        obj_anno = img_anno[1][0]
        cur_ann = 0
        height, width = 512, 512
        for obj in obj_anno:
            obj_name = str(obj[0][0])
            obj_class_id = obj[1][0][0]
            obj_mask = obj[2]
            part_anno = obj[3]
            if len(part_anno) < 1:
                continue
            part_anno = part_anno[0]
            part_idset = objname2partidset[obj_name]
            part_nameset = [partid2partname[part_id].split(':')[1] for part_id in part_idset]
            for part in part_anno:
                part_name = str(part[0][0])
                part_mask = part[1]
                height, width = part_mask.shape
                for part_id_use, part_name_use in zip(part_idset, part_nameset):
                    if part_name_use in part_name:
                        part_num[part_id_use] += 1
                        segmentation, box = mask2json(part_mask)
                        ann_info = {
                            'id': json_ann_id,
                            # 'segmentation': segmentation,
                            'bbox': box,
                            'category_id': part_id_use,
                            'image_id': int(img_path[:4] + img_path[5:]),
                            'iscrowd': 0,
                            'area': box[2] * box[3],
                        }
                        anns.append(ann_info)
                        json_ann_id += 1
                        cur_ann += 1
                        break
        if cur_ann > 0:
            img_info = {
                'id': int(img_path[:4] + img_path[5:]),
                'file_name': img_path + '.jpg',
                'height': height,
                'width': width,
            }
            imgs.append(img_info)

    print(part_num)
    print(len(imgs))
    ann_json['images'] = imgs
    ann_json['annotations'] = anns
    json.dump(ann_json, open(args.ann_out, 'w'))
