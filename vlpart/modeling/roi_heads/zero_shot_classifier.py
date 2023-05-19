# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec


class ZeroShotClassifierGroup(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        zs_weight_path_group: list = ('rand'),
        zs_weight_inference_path: str = 'rand',
        zs_weight_dim: int = 512,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.zs_weight_dim = zs_weight_dim
        self.linear = nn.Linear(input_size, zs_weight_dim)

        for idx, zs_weight_path in enumerate(zs_weight_path_group):
            self.load_from_weight_path(zs_weight_path, 'zs_weight_{}'.format(idx), True, False)

        self.load_from_weight_path(zs_weight_inference_path, 'zs_weight_inference', True, False)

    def load_from_weight_path(self, weight_path, weight_name, concat_bg=True, save_to_pth=False):
        if weight_path.endswith('npy'):
            weight = np.load(weight_path)
            weight = torch.tensor(weight, dtype=torch.float32).permute(1, 0).contiguous()  # dim x C
        elif weight_path.endswith('pth'):
            weight = torch.load(weight_path, map_location='cpu')
            weight = weight.clone().detach().permute(1, 0).contiguous()  # dim x C
        else:
            raise NotImplementedError
        if concat_bg:
            weight = torch.cat(
                [weight, weight.new_zeros((self.zs_weight_dim, 1))], dim=1)  # D x (C + 1)
        if self.norm_weight:
            weight = F.normalize(weight, p=2, dim=0)
        self.register_buffer(weight_name, weight, save_to_pth)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'zs_weight_path_group': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH_GROUP,
            'zs_weight_inference_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
        }

    def forward(self, x, classifier=None, dataset_source=None):
        x = self.linear(x)
        if self.training:
            if classifier is not None:
                zs_weight = classifier.permute(1, 0).contiguous()  # D x C'
                zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                    if self.norm_weight else zs_weight
            else:
                assert dataset_source is not None
                zs_weight = getattr(self, 'zs_weight_{}'.format(dataset_source))
        else:
            zs_weight = self.zs_weight_inference

        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x


class ZeroShotClassifier(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        zs_weight_path: str = 'rand',
        zs_weight_inference_path: str = 'rand',
        zs_weight_dim: int = 512,
        use_bias: float = 0.0,
        norm_weight: bool = True,
        norm_temperature: float = 50.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.norm_weight = norm_weight
        self.norm_temperature = norm_temperature

        self.use_bias = use_bias < 0
        if self.use_bias:
            self.cls_bias = nn.Parameter(torch.ones(1) * use_bias)

        self.zs_weight_dim = zs_weight_dim
        self.linear = nn.Linear(input_size, zs_weight_dim)

        self.load_from_weight_path(zs_weight_path, 'zs_weight', True, True)
        self.load_from_weight_path(zs_weight_inference_path, 'zs_weight_inference', True, False)

    def load_from_weight_path(self, weight_path, weight_name, concat_bg=True, save_to_pth=False):
        if weight_path.endswith('npy'):
            weight = np.load(weight_path)
            weight = torch.tensor(weight, dtype=torch.float32).permute(1, 0).contiguous()  # dim x C
        elif weight_path.endswith('pth'):
            weight = torch.load(weight_path, map_location='cpu')
            weight = weight.clone().detach().permute(1, 0).contiguous()  # dim x C
        else:
            raise NotImplementedError
        if concat_bg:
            weight = torch.cat(
                [weight, weight.new_zeros((self.zs_weight_dim, 1))], dim=1)  # D x (C + 1)
        if self.norm_weight:
            weight = F.normalize(weight, p=2, dim=0)
        self.register_buffer(weight_name, weight, save_to_pth)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'input_shape': input_shape,
            'zs_weight_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH,
            'zs_weight_inference_path': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH,
            'zs_weight_dim': cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_DIM,
            'use_bias': cfg.MODEL.ROI_BOX_HEAD.USE_BIAS,
            'norm_weight': cfg.MODEL.ROI_BOX_HEAD.NORM_WEIGHT,
            'norm_temperature': cfg.MODEL.ROI_BOX_HEAD.NORM_TEMP,
        }

    def forward(self, x, classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x = self.linear(x)
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous()  # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight if self.training else self.zs_weight_inference

        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x
