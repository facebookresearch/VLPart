#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# script to parse imagenet image

# python train_net.py --num-gpus 8 --config-file configs/ann_parser/build_pascalpart.yaml --eval-only
python train_net.py --num-gpus 8 --config-file configs/ann_parser/find_ins_mixer_swinbase.yaml --eval-only \
      MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH 'datasets/metadata/pascal_part_clip_RN50_a+cname.npy' \
      DINO.BASE_OBJ_CAT "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]" \
      DATASETS.TEST "('imagenet_golden_pascal',)" \
      OUTPUT_ANN_DIR 'datasets/imagenet/imagenet_golden_pascal_parsed_swinbase.json'

python train_net.py --num-gpus 8 --config-file configs/ann_parser/find_ins_mixer_swinbase.yaml --eval-only \
      MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH 'datasets/metadata/partimagenet_clip_RN50_a+cname.npy' \
      DINO.BASE_OBJ_CAT "[0,1,2,3,4,5,6,7,8,9,10]" \
      DATASETS.TEST "('imagenet_golden_partimagenet',)" \
      OUTPUT_ANN_DIR 'datasets/imagenet/imagenet_golden_partimagenet_parsed_swinbase.json'

python train_net.py --num-gpus 8 --config-file configs/ann_parser/find_ins_mixer_swinbase.yaml --eval-only \
      MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH 'datasets/metadata/paco_clip_RN50_a+cname.npy' \
      DATASETS.TEST "('imagenet_golden_paco',)" \
      OUTPUT_ANN_DIR 'datasets/imagenet/imagenet_golden_paco_parsed_swinbase.json'

# python train_net.py --num-gpus 8 --config-file configs/ann_parser/find_ins_mixer.yaml --eval-only \
#       DINO.BASE_OBJ_CAT "[]" \
#       DATASETS.TEST "('imagenet_golden_addition',)" \
#       OUTPUT_ANN_DIR 'datasets/imagenet/imagenet_golden_addition_parsed.json'