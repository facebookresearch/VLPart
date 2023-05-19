# Prepare datasets for VLPart

Before preparing datasets, please make sure the models are prepared.

Download the datasets from the official websites and place or sim-link them under `$VLPart_ROOT/datasets/`. 

```
$VLPart_ROOT/datasets/
    metadata/
    partimagenet/
    pascal_part/
    paco/
    coco/
    lvis/
    VOC2007/
    imagenet/
```
`metadata/` is our preprocessed meta-data (included in the repo). See the section [meta](#Metadata) for details.


Please follow the following instruction to pre-process individual datasets.


### PartImageNet

Download PartImageNet images and annotations and from [official repo](https://github.com/TACJu/PartImageNet)

The PartImageNet folder should look like:
```
$VLPart_ROOT/datasets/
    partimagenet/
        train/
            n01440764
            n01443537
            ...            
        val/
            n01484850
            n01614925
            ...
        train.json
        val.json
```

convert them into coco annotation format
~~~
cd $VLPart_ROOT/
python tools/partimagenet_format_json.py --old_path datasets/partimagenet/train.json --new_path datasets/partimagenet/train_format.json
python tools/partimagenet_format_json.py --old_path datasets/partimagenet/val.json --new_path datasets/partimagenet/val_format.json
~~~


### PASCAL Part

Download pascal_part annotations and images from
```
wget http://roozbehm.info/pascal-parts/trainval.tar.gz
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
```

The PASCAL Part folder should look like:
```
$VLPart_ROOT/datasets/
    pascal_part/
        Annotations_Part/
            2008_000002.mat
            2008_000003.mat
            ...
            2010_006086.mat
        VOCdevkit/
            VOC2010/
```

convert them into coco annotation format
~~~
cd $VLPart_ROOT/
python tools/pascal_part_mat2json.py
python tools/pascal_part_mat2json.py --split train.txt --ann_out datasets/pascal_part/train.json
python tools/pascal_part_mat2json.py --only_base --split train.txt --ann_out datasets/pascal_part/train_base.json
python tools/pascal_part_one_json.py
python tools/pascal_part_one_json.py --only_base --part_path datasets/pascal_part/train_base.json --out_path datasets/pascal_part/train_base_one.json
~~~


### PACO

Download paco annotations and images according to [paco](https://github.com/facebookresearch/paco).

The PACO folder should look like:
```
$VLPart_ROOT/datasets/
    paco/
        annotations/
            paco_ego4d_v1_test.json
            paco_ego4d_v1_train.json
            paco_ego4d_v1_val.json
            paco_lvis_v1_test.json
            paco_lvis_v1_train.json
            paco_lvis_v1_val.json
        images/
            000cd456-ff8d-499b-b0c1-4acead128a8b_000024.jpeg
            000cd456-ff8d-499b-b0c1-4acead128a8b_000681.jpeg
            ...    
```


### COCO and LVIS

```
$VLPart_ROOT/datasets/
    coco/
        train2017/
        val2017/
        annotations/
            captions_train2017.json
            instances_train2017.json 
            instances_val2017.json
    lvis/
        lvis_v1_train.json
        lvis_v1_val.json
        lvis_v1_minival_inserted_image_name.json
```
Download lvis_v1_minival_inserted_image_name.json from
```
wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/coco/annotations/lvis_v1_minival_inserted_image_name.json
```

### VOC
Download VOC2007 from
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
```

The VOC2007 folder is in `VOCdevkit/VOC2007`.


### ImageNet

The imagenet folder should look like:
```
$VLPart_ROOT/datasets/
    imagenet/
        train/
            n01440764/
            n01443537/
            ...
```
Collect IN-S11(partimagenet), IN-S20(voc) and IN(golden): 
~~~
cd $VLPart_ROOT/
python tools/partimagenet_supercat_image_info.py
python tools/voc_from_imagenet.py
bash tools/golden_image_info.sh
~~~


### Metadata

```
$VLPart_ROOT/datasets/
    metadata/
        coco_clip_RN50_a+cname.npy
        lvis_v1_clip_RN50_a+cname.npy
        ...
```

`*_clip_a+cname.npy` is the pre-computed CLIP embeddings for each dataset.
They are created by (taking LVIS as an example)
~~~
python tools/lvis_clip_name.py
~~~
