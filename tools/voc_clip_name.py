# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import torch
import numpy as np

voc_categories_fix = [
  {'id': 1, 'name': 'aeroplane'},
  {'id': 2, 'name': 'bicycle'},
  {'id': 3, 'name': 'bird'},
  {'id': 4, 'name': 'boat'},
  {'id': 5, 'name': 'bottle'},
  {'id': 6, 'name': 'bus'},
  {'id': 7, 'name': 'car'},
  {'id': 8, 'name': 'cat'},
  {'id': 9, 'name': 'chair'},
  {'id': 10, 'name': 'cow'},
  {'id': 11, 'name': 'dining table'},
  {'id': 12, 'name': 'dog'},
  {'id': 13, 'name': 'horse'},
  {'id': 14, 'name': 'motorbike'},
  {'id': 15, 'name': 'person'},
  {'id': 16, 'name': 'potted plant'},
  {'id': 17, 'name': 'sheep'},
  {'id': 18, 'name': 'sofa'},
  {'id': 19, 'name': 'train'},
  {'id': 20, 'name': 'tv monitor'},
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', default='datasets/metadata/voc_clip_RN50_a+cname.npy')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="RN50")
    args = parser.parse_args()

    cat_names = [x['name'] for x in sorted(voc_categories_fix, key=lambda x: x['id'])]
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
    text_features = text_features.cpu().numpy()

    if args.out_path != '':
        print('Saving to', args.out_path)
        np.save(args.out_path, text_features)
