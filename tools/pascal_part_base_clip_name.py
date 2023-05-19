# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import torch
import numpy as np


PASCAL_PART_BASE_CATEGORIES = [
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
    {"id": 19, "name": "car:license plate", "abbr": "car:liplate"},
    {"id": 20, "name": "car:headlight"},
    {"id": 21, "name": "car:door"},
    {"id": 22, "name": "car:mirror"},
    {"id": 23, "name": "car:window"},
    {"id": 24, "name": "car:wheel"},
    {"id": 25, "name": "cat:head"},
    {"id": 26, "name": "cat:leg"},
    {"id": 27, "name": "cat:ear"},
    {"id": 28, "name": "cat:eye"},
    {"id": 29, "name": "cat:paw", "abbr": "cat:pa"},
    {"id": 30, "name": "cat:neck"},
    {"id": 31, "name": "cat:nose"},
    {"id": 32, "name": "cat:tail"},
    {"id": 33, "name": "cat:torso"},
    {"id": 34, "name": "cow:head"},
    {"id": 35, "name": "cow:leg"},
    {"id": 36, "name": "cow:ear"},
    {"id": 37, "name": "cow:eye"},
    {"id": 38, "name": "cow:neck"},
    {"id": 39, "name": "cow:horn"},
    {"id": 40, "name": "cow:muzzle"},
    {"id": 41, "name": "cow:tail"},
    {"id": 42, "name": "cow:torso"},
    {"id": 43, "name": "horse:head"},
    {"id": 44, "name": "horse:leg"},
    {"id": 45, "name": "horse:ear"},
    {"id": 46, "name": "horse:eye"},
    {"id": 47, "name": "horse:neck"},
    {"id": 48, "name": "horse:muzzle"},
    {"id": 49, "name": "horse:tail"},
    {"id": 50, "name": "horse:torso"},
    {"id": 51, "name": "motorbike:wheel"},
    {"id": 52, "name": "motorbike:handlebar"},
    {"id": 53, "name": "motorbike:headlight"},
    {"id": 54, "name": "motorbike:saddle"},
    {"id": 55, "name": "person:hair"},
    {"id": 56, "name": "person:head"},
    {"id": 57, "name": "person:ear"},
    {"id": 58, "name": "person:eye"},
    {"id": 59, "name": "person:nose"},
    {"id": 60, "name": "person:neck"},
    {"id": 61, "name": "person:mouth"},
    {"id": 62, "name": "person:arm"},
    {"id": 63, "name": "person:hand"},
    {"id": 64, "name": "person:leg"},
    {"id": 65, "name": "person:foot"},
    {"id": 66, "name": "person:torso"},
    {"id": 67, "name": "potted plant:plant"},
    {"id": 68, "name": "potted plant:pot"},
    {"id": 69, "name": "sheep:head"},
    {"id": 70, "name": "sheep:leg"},
    {"id": 71, "name": "sheep:ear"},
    {"id": 72, "name": "sheep:eye"},
    {"id": 73, "name": "sheep:neck"},
    {"id": 74, "name": "sheep:horn"},
    {"id": 75, "name": "sheep:muzzle"},
    {"id": 76, "name": "sheep:tail"},
    {"id": 77, "name": "sheep:torso"},
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='datasets/metadata/pascal_part_base_clip_RN50_a+cname.npy')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="RN50")
    parser.add_argument('--use_part', default=True, action='store_true')
    parser.add_argument('--only_base', action='store_true')
    args = parser.parse_args()

    cat_names = [x['name'].lower().strip() for x in \
                 sorted(PASCAL_PART_BASE_CATEGORIES, key=lambda x: x['id'])]

    sentences = []
    if args.use_part:
        cat_names = [x.replace(':', ' ') for x in cat_names]
        print('cat_names', cat_names)
        sentences = [x for x in cat_names]

    if args.prompt == 'a':
        sentences = ['a ' + x for x in cat_names]
    if args.prompt == 'none':
        sentences = [x for x in cat_names]
    elif args.prompt == 'photo':
        sentences = ['a photo of a {}'.format(x) for x in cat_names]
    elif args.prompt == 'scene':
        sentences = ['a photo of a {} in the scene'.format(x) for x in cat_names]
    print('sentences', sentences)

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    text_features = text_features.cpu().numpy()

    if args.out_path != '':
        print('Saving to', args.out_path)
        np.save(args.out_path, text_features)
