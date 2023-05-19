# VLPart model zoo

This file documents a collection of models reported in our paper.
The training time was measured on with 8 NVIDIA V100 GPUs & NVLink.

#### How to Read the Tables

The "Name" column contains a link to the config file. 

To train a model, run:

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml
``` 

To evaluate a model with a trained/pretrained model, run:

```
python train_net.py --num-gpus 8 --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth
``` 

An example of cross-dataset evaluation: 
```
python train_net.py --num-gpus 8 --config-file configs/partimagenet/r50_partimagenet.yaml --eval-only MODEL.WEIGHTS models/r50_pascalpart.pth
``` 

Before training, make sure [Preparing Datasets](datasets) and [Preparing Models](models) are well-prepared.

<br>


### Cross-dataset part segmentation on PartImageNet

| Config                                                                                 | All(40) AP | quad-: head | quad-: body | quad-: foot | quad-: tail | Training time |                                                Download                                                 |
|----------------------------------------------------------------------------------------|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-------------:|:-------------------------------------------------------------------------------------------------------:|
| [pascal_part](configs/pascal_part/r50_pascalpart.yaml)                                 |    4.5     |    17.4     |     0.1     |     0.0     |     2.9     |      1h       |          [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpart.pth)          |
| [+ IN-S11 label](configs/partimagenet_ablation/r50_pascalpart_ins11.yaml)              |    5.4     |    23.6     |     3.4     |     0.8     |     1.2     |     1.5h      |       [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpart_ins11.pth)       |
| [+ IN-S11 parsed](configs/partimagenet_ablation/r50_pascalpart_ins11_ins11parsed.yaml) |    7.8     |    35.0     |    15.2     |     3.5     |     8.9     |      3h       | [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpart_ins11_ins11parsed.pth) |

| Config                                                                                           | All(40) AP | quad-: head | quad-: body | quad-: foot | quad-: tail | Training time |                                                     Download                                                      |
|--------------------------------------------------------------------------------------------------|:----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-------------:|:-----------------------------------------------------------------------------------------------------------------:|
| [pascal_part](configs/pascal_part/r50_pascalpart.yaml)                                           |    4.5     |    17.4     |     0.1     |     0.0     |     2.9     |      1h       |               [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpart.pth)               |
| [+ LVIS_PACO](configs/partimagenet_ablation/r50_lvis_paco_pascalpart.yaml)                       |    7.8     |    22.9     |     7.1     |     0.3     |     4.0     |  15h + 2.5h   |          [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart.pth)          |
| [+ IN-S11 label](configs/partimagenet_ablation/r50_lvis_paco_pascalpart_ins11.yaml)              |    8.8     |    26.3     |     3.7     |     0.4     |     1.0     |      3h       |       [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart_ins11.pth)       |
| [+ IN-S11 parsed](configs/partimagenet_ablation/r50_lvis_paco_pascalpart_ins11_ins11parsed.yaml) |    11.8    |    47.5     |     13.4    |     4.5     |     14.8    |      3h       | [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart_ins11_ins11parsed.pth) |


- The evaluation metric is mAPmask@[0.5:0.95] on the validation set of PartImageNet.
- pascal_part + LVIS_PACO is training first(15h) on LVIS and PACO [r50_lvis_paco.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_coco.pth), then(2.5h) on LVIS, PACO and Pascal Part.
- Before training on IN-S11 parsed, generate IN-S11 parsed(20min) by:
```
python train_net.py --num-gpus 8 --config-file configs/ann_parser/build_pascalpart.yaml --eval-only
python train_net.py --num-gpus 8 --config-file configs/ann_parser/find_ins11_mixer.yaml --eval-only 
```
or download [partimagenet_parsed.json](https://github.com/PeizeSun/VLPart/releases/download/v0.1/partimagenet_parsed.json) and put it to `$VLPart_ROOT/datasets/partimagenet/`.

<br>


### Cross-category part segmentation within Pascal Part

| Config                                                                                        | All(93) AP/AP50 | Base(77) AP/AP50 | Novel(16) AP/AP50 | dog: head | dog: torso | dog: leg | dog: paw | dog: tail | Training time |                                                      Download                                                     |
|-----------------------------------------------------------------------------------------------|:---------------:|:----------------:|:-----------------:|:---------:|:----------:|:--------:|:--------:|:---------:|:-------------:|:-----------------------------------------------------------------------------------------------------------------:|
| [pascal_part_base](configs/pascal_part_ablation/r50_pascalpartbase.yaml)                      |    15.0/33.4    |    17.8/39.6     |      1.5/3.7      |    6.1    |    7.9     |   2.9    |   13.8   |    3.2    |      1h       |              [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpartbase.pth)            |
| [+ VOC object](configs/pascal_part_ablation/r50_pascalpartbase_voc.yaml)                      |    16.8/36.8    |    19.9/43.3     |      2.1/5.9      |   29.9    |    22.6    |   3.2    |   12.4   |    2.1    |     1.5h      |            [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpartbase_voc.pth)          |
| [+ IN-S20 label](configs/pascal_part_ablation/r50_pascalpartbase_voc_ins20.yaml)              |    17.4/37.5    |    20.8/44.7     |      1.1/3.1      |   12.8    |    17.8    |   2.0    |   5.9    |    0.9    |      3h       |         [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpartbase_voc_ins20.pth)       |
| [+ IN-S20 parsed](configs/pascal_part_ablation/r50_pascalpartbase_voc_ins20_ins20parsed.yaml) |    18.4/39.4    |    21.3/45.3     |     4.2/11.0      |   28.7    |    34.8    |   17.2   |   5.7    |   14.3    |     4.5h      |   [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpartbase_voc_ins20_ins20parsed.pth) |

- The evaluation metric is APmask@0.5 on the validation set of Pascal Part.
- Before training on IN-S20 parsed, generate IN-S20 parsed(50min) by:
```
python train_net.py --num-gpus 8 --config-file configs/ann_parser/build_pascalpartbase.yaml --eval-only
python train_net.py --num-gpus 8 --config-file configs/ann_parser/find_ins20_mixer.yaml --eval-only 
```
or download [imagenet_voc_image_parsed.json](https://github.com/PeizeSun/VLPart/releases/download/v0.1/imagenet_voc_image_parsed.json) and put it to `$VLPart_ROOT/datasets/imagenet/`.

<br>


### Open-vocabulary object detection and part segmentation

R50 Mask R-CNN:

| Name               |                                     VOC AP/AP50                                      |                                      COCO AP/AP50                                      |                                      LVIS AP/APr                                       |                                          PartImageNet AP/AP50                                          |                                        Pascal Part AP/AP50                                         |                                      PACO AP/AP50                                      |
|--------------------|:------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|
| Dataset-specific   |                                      35.9/69.7                                       |                                       38.0/60.8                                        |                                       28.1/20.8                                        |                                               29.7/54.1                                                |                                             19.4/42.3                                              |                                       10.6/21.7                                        |
| Config             |                         [r50_voc](configs/voc/r50_voc.yaml)                          |                         [r50_coco](configs/coco/r50_coco.yaml)                         |                         [r50_lvis](configs/lvis/r50_lvis.yaml)                         |                     [r50_partimagenet](configs/partimagenet/r50_partimagenet.yaml)                     |                     [r50_pascalpart](configs/pascal_part/r50_pascalpart.yaml)                      |                         [r50_paco](configs/paco/r50_paco.yaml)                         |   
| Training Time      |                                          2h                                          |                                          6.5h                                          |                                           7h                                           |                                                   2h                                                   |                                                 1h                                                 |                                          7h                                            |
| Download           | [r50_voc.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_voc.pth) | [r50_coco.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_coco.pth) | [r50_lvis.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis.pth) | [r50_partimagenet.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_partimagenet.pth) | [r50_pascalpart.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpart.pth) | [r50_paco.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_paco.pth) |


| Config                                                                             | VOC AP/AP50 | COCO AP/AP50 | LVIS AP/APr | PartImageNet AP/AP50 | Pascal Part AP/AP50 | PACO AP/AP50 | Training time |                                                       Download                                                        |
|------------------------------------------------------------------------------------|:-----------:|:------------:|:-----------:|:--------------------:|:-------------------:|:------------:|:-------------:|:---------------------------------------------------------------------------------------------------------------------:|
| [joint](configs/joint/r50_lvis_paco.yaml)                                          |  44.5/70.3  |  29.0/48.1   |  27.3/19.0  |       5.4/11.3       |      4.9/11.3       |   9.6/19.5   |      15h      |                 [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco.pth)                  |
| [joint*](configs/joint/r50_lvis_paco_pascalpart.yaml)                              |  42.8/70.8  |  28.6/48.0   |  26.8/20.4  |       7.8/15.3       |      21.6/46.3      |   9.3/18.9   |  15h + 2.5h   |            [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart.pth)            |
| [joint**](configs/joint/r50_lvis_paco_pascalpart_partimagenet.yaml)                |  40.6/69.3  |  28.4/47.8   |  26.4/16.0  |      29.1/52.0       |      22.6/47.8      |   9.3/18.9   |   15h + 3h    |     [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart_partimagenet.pth)      |
| [+ IN label](configs/joint_in/r50_lvis_paco_pascalpart_partimagenet_in.yaml)       |  38.0/67.8  |  28.2/47.8   |  26.0/15.9  |      30.8/54.4       |      23.6/49.2      |   9.0/18.7   | 15h + 3h + 4h |    [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart_partimagenet_in.pth)    |
| [+ IN parsed](configs/joint_in/r50_lvis_paco_pascalpart_partimagenet_inparsed.yaml) |  38.3/67.8  |  28.5/47.8   |  26.2/17.8  |      31.6/55.7       |      24.0/49.8      |   9.6/20.2   | 15h + 3h + 6h | [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart_partimagenet_inparsed.pth) |


- joint is training on LVIS and PACO.
- joint* is training first(15h) on LVIS and PACO, then(2.5h) on LVIS, PACO, Pascal Part.
- joint** is training first(15h) on LVIS and PACO, then(3h) on LVIS, PACO, Pascal Part, PartImageNet.
- Before training on IN parsed, generate IN parsed(100min) by:
```
bash tools/golden_image_parse.sh
```
or download [golden_image_parsed.zip](https://github.com/PeizeSun/VLPart/releases/download/v0.1/golden_image_parsed.zip), put it to `$VLPart_ROOT/datasets/imagenet/` and unzip it.
 

<br>

SwinBase Cascade Mask R-CNN:

| Name              |                                              VOC AP/AP50                                               |                                               COCO AP/AP50                                               |                                               LVIS AP/APr                                                |                                                   PartImageNet AP/AP50                                                   |                                                 Pascal Part AP/AP50                                                  |                                               PACO AP/AP50                                               |
|-------------------|:------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------:|
| Dataset-specific  |                                               59.0/82.0                                                |                                                52.5/72.0                                                 |                                                43.1/38.7                                                 |                                                        41.7/68.7                                                         |                                                      27.4/56.1                                                       |                                                15.2/29.4                                                 |
| Config            |                         [swinbase_voc](configs/coco/swinbase_cascade_voc.yaml)                         |                         [swinbase_coco](configs/coco/swinbase_cascade_coco.yaml)                         |                         [swinbase_lvis](configs/lvis/swinbase_cascade_lvis.yaml)                         |                     [swinbase_partimagenet](configs/partimagenet/swinbase_cascade_partimagenet.yaml)                     |                     [swinbase_pascal_part](configs/pascal_part/swinbase_cascade_pascalpart.yaml)                     |                         [swinbase_paco](configs/paco/swinbase_cascade_paco.yaml)                         |   
| Training Time     |                                                   4h                                                   |                                                 1day15h                                                  |                                                 1day15h                                                  |                             4.5h                                                                                         |                                                        1.5h                                                          |                                                  1day2h                                                  |
| Download          | [swinbase_voc.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_voc.pth) | [swinbase_coco.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_coco.pth) | [swinbase_lvis.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis.pth) | [swinbase_partimagenet.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_partimagenet.pth) | [swinbase_pascalpart.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_pascalpart.pth) | [swinbase_paco.pth](https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_paco.pth) |


| Config                                                                                           | VOC AP/AP50 | COCO AP/AP50 | LVIS AP/APr | PartImageNet AP/AP50 | Pascal Part AP/AP50 | PACO AP/AP50 |   Training time   |                                                                    Download                                                                     |
|--------------------------------------------------------------------------------------------------|:-----------:|:------------:|:-----------:|:--------------------:|:-------------------:|:------------:|:-----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------:|
| [joint](configs/joint/swinbase_cascade_lvis_paco.yaml)                                           |  55.2/72.2  |  41.0/58.4   |  41.3/32.8  |       6.9/13.7       |      5.6/12.5       |  15.9/31.9   |      2day5h       |                        [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco.pth)                        |
| [joint*](configs/joint/swinbase_cascade_lvis_paco_pascalpart.yaml)                               |  52.6/72.4  |  40.4/57.9   |  39.9/29.8  |      11.8/21.8       |      30.5/59.3      |  15.4/30.2   |   2day5h + 4.5h   |                  [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart.pth)                   |
| [joint**](configs/joint/swinbase_cascade_lvis_paco_pascalpart_partimagenet.yaml)                 |  50.3/71.6  |  40.3/57.8   |  39.6/30.3  |      40.0/64.8       |      31.2/60.5      |  15.4/30.3   |    2day5h + 6h    |            [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart_partimagenet.pth)            | 
| [+ IN label](configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_in.yaml)        |  48.1/69.7  |  40.3/57.7   |  39.3/28.9  |      41.2/66.8       |      31.7/61.1      |  15.9/30.8   | 2day5h + 6h + 8h  |          [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart_partimagenet_in.pth)           |
| [+ IN parsed](configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.yaml) |  47.8/69.7  |  40.5/58.1   |  39.6/30.5  |      42.0/68.2       |      31.9/61.6      |  15.6/30.6   | 2day5h + 6h + 20h |       [model](https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth)        |


- joint is training on LVIS and PACO.
- joint* is training first(2day5h) on LVIS and PACO, then(4.5h) on LVIS, PACO, Pascal Part.
- joint** is training first(2day5h) on LVIS and PACO, then(6h) on LVIS, PACO, Pascal Part, PartImageNet.
- Before training on IN parsed, generate IN parsed(70min) by:
```
bash tools/golden_image_parse_swinbase.sh
```
or download [golden_image_parsed_swinbase.zip](https://github.com/PeizeSun/VLPart/releases/download/v0.1/golden_image_parsed_swinbase.zip), put it to `$VLPart_ROOT/datasets/imagenet/` and unzip it.
 
