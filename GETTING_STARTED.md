## Getting Started

This document provides a brief intro of the usage of VLPart.



### Inference Demo on Image

Pick a model and its config file from [Model Zoo](MODEL_ZOO.md), run:
```
python demo/demo.py --config-file configs/pascal_part/r50_pascalpart.yaml \
  --input input1.jpg input2.jpg \
  --output output_image \
  --vocabulary pascal_part \
  --confidence-threshold 0.7 \
  --opts MODEL.WEIGHTS /path/to/checkpoint_file VIS.BOX False
```
With no need to change `config-file`, `--vocabulary` can be `['pascal_part', 'partimagenet', 'paco', 'voc', 'coco', 'lvis', 'pascal_part_voc', 'lvis_paco',]`


Our model also supports `custom vocabulary`, for example:
```
python demo/demo.py --config-file configs/pascal_part/r50_pascalpart.yaml \
  --input input1.jpg input2.jpg \
  --output output_image \
  --vocabulary custom \
  --custom_vocabulary "fox head,fox torso,fox leg" \
  --confidence-threshold 0.7 \
  --opts MODEL.WEIGHTS /path/to/checkpoint_file VIS.BOX False
```



### Training & Evaluation in Command Line

We provide `train_net.py` to train all the configs provided.

```
python train_net.py --num-gpus 8 \
  --config-file configs/pascal_part/r50_pascalpart.yaml
```

The configs are made for 8-GPU training.
To train on 1 GPU, you need to figure out learning rate, batch size and training iterations by yourself, 
as it has not been checked to reproduce performance.
```
python train_net.py \
  --config-file configs/pascal_part/r50_pascalpart.yaml --num-gpus 1 \
  SOLVER.IMS_PER_BATCH REASONABLE_VALUE SOLVER.BASE_LR REASONABLE_VALUE SOLVER.STEPS "(REASONABLE_VALUE,)"
```

To evaluate a model's performance, use:
```
python train_net.py --num-gpus 8 \
  --config-file configs/pascal_part/r50_pascalpart.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

More detailed command lines are in [Model Zoo](./MODEL_ZOO.md).
