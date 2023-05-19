## Third Party Methods on Part Detection

This document provides the steps of applying third party methods on part detection.

### Peformance

The evaluation metric is APbox on validation set of PASCAL Part.

| Method                                                | dog: head | dog: leg | dog: paw | dog: tail | dog: torso | 
|-------------------------------------------------------|:---------:|:--------:|:--------:|:---------:|:----------:|
| [RegionCLIP](https://github.com/microsoft/RegionCLIP) |    5.2    |   0.1    |   0.2    |    0.0    |    1.9     | 
| [Detic](https://github.com/facebookresearch/Detic)    |    3.2    |   0.0    |   0.0    |    0.0    |    2.0     |
| [VLDet](https://github.com/clin1223/VLDet)            |    3.5    |   0.0    |   0.0    |    0.0    |    1.9     |
| [GLIP](https://github.com/microsoft/GLIP)             |   32.6    |   3.1    |   2.7    |    2.2    |    10.5    |


### RegionCLIP
1. Git clone and install [RegionCLIP](https://github.com/microsoft/RegionCLIP)
2. Prepare the folder [pretrained_ckpt](https://drive.google.com/drive/folders/1hzrJBvcCrahoRcqJRqzkIGFO_HUSJIii) 
3. Link the dataset to RegionCLIP
```
$RegionCLIP_ROOT/datasets/pascal_part/
```
4. Add these files to RegionCLIP
```
$RegionCLIP_ROOT/detectron2/data/datasets/pascalpart.py
$RegionCLIP_ROOT/detectron2/data/datasets/buildin.py
$RegionCLIP_ROOT/configs/CLIP_fast_rcnn_R_50_C4_part.yaml
$RegionCLIP_ROOT/tools/dump_clip_features_pascal_part.py
$RegionCLIP_ROOT/test_zeroshot_inference_part.sh
```
5. Generate the classifier
```
cd $RegionCLIP_ROOT
python tools/dump_clip_features_pascal_part.py
```
6. Evaluate on Pascal Part 
```
cd $RegionCLIP_ROOT
sh test_zeroshot_inference_part.sh
```


### Detic
1. Git clone and install [Detic](https://github.com/facebookresearch/Detic)
2. Download model weight [Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth](https://dl.fbaipublicfiles.com/detic/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth)
3. Link the dataset to Detic
```
$Detic_ROOT/datasets/pascal_part/
```
4. Add these files to Detic
```
$Detic_ROOT/detic/data/datasets/pascalpart.py
$Detic_ROOT/detic/__init__.py
$Detic_ROOT/configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size_part.yaml
$Detic_ROOT/tools/dump_clip_features_pascal_part.py
```
5. Generate the classifier
```
cd $Detic_ROOT
python tools/dump_clip_features_pascal_part.py
```
6. Evaluate on Pascal Part 
```
cd $Detic_ROOT
python train_net.py --config-file configs/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size_part.yaml --num-gpus 4 --eval-only MODEL.WEIGHTS Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

### VLDet
1. Git clone and install [VLDet](https://github.com/clin1223/VLDet)
2. Download model weight [lvis_vldet_swinB.pth](https://drive.google.com/drive/folders/1ngb1mBOUvFpkcUM7D3bgIkMdUj2W5FUa)
3. Link the dataset to VLDet
```
$VLDet_ROOT/datasets/pascal_part/
```
4. Add these files to VLDet
```
$VLDet_ROOT/vldet/data/datasets/pascalpart.py
$VLDet_ROOT/vldet/data/datasets/coco_zeroshot.py
$VLDet_ROOT/vldet/__init__.py
$VLDet_ROOT/configs/VLDet_LbaseI_CLIP_SwinB_896b32_2x_ft4x_caption_part.yaml
$VLDet_ROOT/tools/dump_clip_features_pascal_part.py
```
5. Generate the classifier
```
cd $VLDet_ROOT
python tools/dump_clip_features_pascal_part.py
```
6. Evaluate on Pascal Part
```
cd $VLDet_ROOT
python train_net.py --config-file configs/VLDet_LbaseI_CLIP_SwinB_896b32_2x_ft4x_caption_part.yaml --num-gpus 4 --eval-only MODEL.WEIGHTS lvis_vldet_swinB.pth
```

### GLIP
1. Git clone and install [GLIP](https://github.com/microsoft/GLIP)
2. Download model weight [glip_tiny_model_o365_goldg_cc_sbu.pth](https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth)
3. Link the dataset to GLIP
```
$GLIP_ROOT/datasets/pascal_part/
```
4. Add these files to GLIP
```
$GLIP_ROOT/pascal_part_mat2json_start1.py
$GLIP_ROOT/part_task_config.yaml
$GLIP_ROOT/inference_part.sh
```
5. Generate Pascal Part Dog val json
```
cd $GLIP_ROOT
python pascal_part_mat2json_start1.py
```
6. Evaluate on Pascal Part Dog
```
cd $GLIP_ROOT
sh inference_part.sh
```
The per category performance is in output folder.