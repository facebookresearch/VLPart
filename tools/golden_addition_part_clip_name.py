# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import numpy as np
import json
import torch
import random
import sys
sys.path.append('.')
from vlpart.data.datasets.golden_part import (
    ADDITIONAL_PART_CATEGORIES,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='datasets/metadata/addition_part_clip_RN50_a+cname.npy')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="RN50")
    parser.add_argument('--use_part', default=True, action='store_true')
    args = parser.parse_args()

    golden_categories = ADDITIONAL_PART_CATEGORIES

    cat_names = [x['name'].lower().strip() for x in \
                 sorted(golden_categories, key=lambda x: x['id'])]
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