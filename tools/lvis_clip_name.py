# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import torch
import numpy as np
import itertools
from nltk.corpus import wordnet
import sys
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES


def map_name(x):
    x = x.replace('_', ' ')
    if '(' in x:
        x = x[:x.find('(')]
    return x.lower().strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/lvis/lvis_v1_val.json')
    parser.add_argument('--out_path', default='datasets/metadata/lvis_v1_clip_RN50_a+cname.npy')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="RN50")
    parser.add_argument('--fix_space', action='store_true')
    parser.add_argument('--use_underscore', action='store_true')
    parser.add_argument('--avg_synonyms', action='store_true')
    parser.add_argument('--use_wn_name', action='store_true')
    parser.add_argument('--map_name', action='store_true')
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cat_names = [x['name'] for x in sorted(data['categories'], key=lambda x: x['id'])]
    if 'synonyms' in data['categories'][0]:
        if args.use_wn_name:
            synonyms = [
                [xx.name() for xx in wordnet.synset(x['synset']).lemmas()] \
                    if x['synset'] != 'stop_sign.n.01' else ['stop_sign'] \
                for x in sorted(data['categories'], key=lambda x: x['id'])]
        else:
            synonyms = [x['synonyms'] for x in \
                        sorted(data['categories'], key=lambda x: x['id'])]
    else:
        synonyms = []

    if args.fix_space:
        cat_names = [x.replace('_', ' ') for x in cat_names]
    if args.use_underscore:
        cat_names = [x.strip().replace('/ ', '/').replace(' ', '_') for x in cat_names]
    if args.map_name:
        cat_names = [map_name(x) for x in cat_names]

    print('cat_names', cat_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sentences, sentences_synonyms = [], []
    if args.prompt == 'a':
        sentences = ['a ' + x for x in cat_names]
        sentences_synonyms = [['a ' + xx for xx in x] for x in synonyms]
    if args.prompt == 'none':
        sentences = [x for x in cat_names]
        sentences_synonyms = [[xx for xx in x] for x in synonyms]
    elif args.prompt == 'photo':
        sentences = ['a photo of a {}'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {}'.format(xx) for xx in x] \
                              for x in synonyms]
    elif args.prompt == 'scene':
        sentences = ['a photo of a {} in the scene'.format(x) for x in cat_names]
        sentences_synonyms = [['a photo of a {} in the scene'.format(xx) for xx in x] \
                              for x in synonyms]

    print('sentences_synonyms', len(sentences_synonyms), \
          sum(len(x) for x in sentences_synonyms))

    import clip
    print('Loading CLIP')
    model, preprocess = clip.load(args.clip_model, device=device)
    if args.avg_synonyms:
        sentences = list(itertools.chain.from_iterable(sentences_synonyms))
        print('flattened_sentences', len(sentences))
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
    if args.avg_synonyms:
        synonyms_per_cat = [len(x) for x in sentences_synonyms]
        text_features = text_features.split(synonyms_per_cat, dim=0)
        text_features = [x.mean(dim=0) for x in text_features]
        text_features = torch.stack(text_features, dim=0)
        print('after stack', text_features.shape)
    text_features = text_features.cpu().numpy()

    if args.out_path != '':
        print('Saving to', args.out_path)
        np.save(args.out_path, text_features)