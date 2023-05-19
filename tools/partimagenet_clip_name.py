# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import torch
import numpy as np

PARTIMAGENET_CATEGORIES = [
    {"id": 0, "name": "Quadruped Head"},
    {"id": 1, "name": "Quadruped Body"},
    {"id": 2, "name": "Quadruped Foot"},
    {"id": 3, "name": "Quadruped Tail"},
    {"id": 4, "name": "Biped Head"},
    {"id": 5, "name": "Biped Body"},
    {"id": 6, "name": "Biped Hand"},
    {"id": 7, "name": "Biped Foot"},
    {"id": 8, "name": "Biped Tail"},
    {"id": 9, "name": "Fish Head"},
    {"id": 10, "name": "Fish Body"},
    {"id": 11, "name": "Fish Fin"},
    {"id": 12, "name": "Fish Tail"},
    {"id": 13, "name": "Bird Head"},
    {"id": 14, "name": "Bird Body"},
    {"id": 15, "name": "Bird Wing"},
    {"id": 16, "name": "Bird Foot"},
    {"id": 17, "name": "Bird Tail"},
    {"id": 18, "name": "Snake Head"},
    {"id": 19, "name": "Snake Body"},
    {"id": 20, "name": "Reptile Head"},
    {"id": 21, "name": "Reptile Body"},
    {"id": 22, "name": "Reptile Foot"},
    {"id": 23, "name": "Reptile Tail"},
    {"id": 24, "name": "Car Body"},
    {"id": 25, "name": "Car Tier"},
    {"id": 26, "name": "Car Side Mirror"},
    {"id": 27, "name": "Bicycle Body"},
    {"id": 28, "name": "Bicycle Head"},
    {"id": 29, "name": "Bicycle Seat"},
    {"id": 30, "name": "Bicycle Tier"},
    {"id": 31, "name": "Boat Body"},
    {"id": 32, "name": "Boat Sail"},
    {"id": 33, "name": "Aeroplane Head"},
    {"id": 34, "name": "Aeroplane Body"},
    {"id": 35, "name": "Aeroplane Engine"},
    {"id": 36, "name": "Aeroplane Wing"},
    {"id": 37, "name": "Aeroplane Tail"},
    {"id": 38, "name": "Bottle Mouth"},
    {"id": 39, "name": "Bottle Body"},
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='datasets/metadata/partimagenet_clip_RN50_a+cname.npy')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="RN50")
    args = parser.parse_args()

    cat_names = [x['name'].lower().strip() for x in \
                 sorted(PARTIMAGENET_CATEGORIES, key=lambda x: x['id'])]
    print('cat_names', cat_names)

    sentences = []
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
