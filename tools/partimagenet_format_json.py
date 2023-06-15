# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_path', default='datasets/partimagenet/train.json')
    parser.add_argument('--new_path', default='datasets/partimagenet/train_format.json')
    args = parser.parse_args()

    print('Loading PartImageNet json')
    data = json.load(open(args.old_path, 'r'))

    for image in data['images']:
        file_name = image['file_name']
        new_file_name = file_name[:9] +'/'+ file_name
        image['file_name'] = new_file_name

    for ann in data['annotations']:
        segs = ann['segmentation']
        new_segs = []
        for seg in segs:
            assert len(seg) > 0 and len(seg) % 2 == 0
            if len(seg) < 4:
                new_segs.append(seg + [0, 0, seg[0], seg[1]])
            if len(seg) == 4:
                new_segs.append(seg + [seg[0], seg[1]])
            else:
                new_segs.append(seg)
        ann['segmentation'] = new_segs

    print('Writing to', args.new_path)
    json.dump(data, open(args.new_path, 'w'))
