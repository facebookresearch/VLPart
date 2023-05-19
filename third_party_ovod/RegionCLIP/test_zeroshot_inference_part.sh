# Copyright (c) Facebook, Inc. and its affiliates.
python3 ./tools/train_net.py \
--eval-only  \
--num-gpus 4 \
--config-file ./configs/CLIP_fast_rcnn_R_50_C4_part.yaml \
MODEL.WEIGHTS ./pretrained_ckpt/regionclip/regionclip_finetuned-lvis_rn50.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG ./configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.BB_RPN_WEIGHTS ./pretrained_ckpt/rpn/rpn_lvis_866.pth \
MODEL.CLIP.TEXT_EMB_PATH pascal_part_clip_a+cname.pth \
MODEL.CLIP.OPENSET_TEST_TEXT_EMB_PATH pascal_part_clip_a+cname.pth \
MODEL.CLIP.CROP_REGION_TYPE RPN \
MODEL.CLIP.MULTIPLY_RPN_SCORE True \
