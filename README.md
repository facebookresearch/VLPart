# Going Denser with Open-Vocabulary Part Segmentation

![](docs/boom.png)

Object detection has been expanded from a limited number of categories to open vocabulary. 
Moving forward, a complete intelligent vision system requires understanding more fine-grained object descriptions, object parts. 
In this work, we propose a detector with the ability to predict both open-vocabulary objects and their part segmentation.
This ability comes from two designs:
- We train the detector on the joint of part-level, object-level and image-level data. 
- We parse the novel object into its parts by its dense semantic correspondence with the base object.

[[`arXiv`](https://arxiv.org/abs/2305.11173)]

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets](datasets) and [Preparing Models](models).

See [Getting Started](GETTING_STARTED.md) for demo, training and inference.



## Model Zoo

We provide a large set of baseline results and trained models in the [Model Zoo](MODEL_ZOO.md).



## License

The majority of this project is licensed under a [MIT License](LICENSE). Portions of the project are available under separate license of referred projects, including [CLIP](https://github.com/openai/CLIP), [Detic](https://github.com/facebookresearch/Detic) and [dino-vit-features](https://github.com/ShirAmir/dino-vit-features). Many thanks for their wonderful works.


## Citation

If you use VLPart in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:

```BibTeX

@article{peize2023vlpart,
  title   =  {Going Denser with Open-Vocabulary Part Segmentation},
  author  =  {Sun, Peize and Chen, Shoufa and Zhu, Chenchen and Xiao, Fanyi and Luo, Ping and Xie, Saining and Yan, Zhicheng},
  journal =  {arXiv preprint arXiv:2305.11173},
  year    =  {2023}
}
```

## :fire:  Extension Project

[Grounded Segment Anything: From Objects to Parts](https://github.com/Cheems-Seminar/grounded-segment-any-parts): A dialogue system to detect, segment and edit anything in part-level in the image.

[Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM): A universal image segmentation model to enable segment and recognize anything at any desired granularity.
