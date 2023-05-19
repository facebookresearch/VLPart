# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import torch
import numpy as np

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='pascal_part_clip_a+cname.pth')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="RN50")
    parser.add_argument('--use_part', default=True, action='store_true')
    parser.add_argument('--only_dog', action='store_true')
    args = parser.parse_args()

    PART_CATEGORIES = PASCAL_PART_DOG_CATEGORIES if args.only_dog else PASCAL_PART_CATEGORIES

    cat_names = [x['name'].lower().strip() for x in \
                 sorted(PART_CATEGORIES, key=lambda x: x['id'])]
    if args.use_part:
        cat_names = [x.replace(':', ' ') for x in cat_names]
    print('cat_names', cat_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentences = []
    if args.prompt == 'a':
        sentences = ['a ' + x for x in cat_names]
    if args.prompt == 'none':
        sentences = [x for x in cat_names]
    elif args.prompt == 'photo':
        sentences = ['a photo of a {}'.format(x) for x in cat_names]
    elif args.prompt == 'scene':
        sentences = ['a photo of a {} in the scene'.format(x) for x in cat_names]

    import clip
    print('Loading CLIP')
    model, preprocess = clip.load(args.clip_model, device=device)
    text = clip.tokenize(sentences).to(device)
    with torch.no_grad():
        if len(text) > 10000:
            text_features = torch.cat([
                model.encode_text(text[:len(text) // 2]),
                model.encode_text(text[len(text) // 2:])],
                dim=0)
        else:
            text_features = model.encode_text(text)
    print('text_features.shape', text_features.shape)
    text_features = text_features.cpu()

    if args.out_path != '':
        print('Saving to', args.out_path)
        torch.save(text_features, args.out_path)
