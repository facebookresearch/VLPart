# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Peize Sun from https://github.com/ShirAmir/dino-vit-features
# The original code is under MIT License
import logging
import os

from PIL import Image
import numpy as np
import itertools
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.modules.utils as nn_utils
import timm
import types
import math
import random
from torchvision import transforms

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like, FrozenBatchNorm2d
from detectron2.structures import ImageList, Instances, BitMasks, ROIMasks, Boxes, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
import detectron2.utils.comm as comm

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from torch.cuda.amp import autocast

from ..text_encoder.text_encoder import build_text_encoder
from ..utils.detic import load_class_freq, get_fed_loss_inds

import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as mask_util


@META_ARCH_REGISTRY.register()
class ParserNetMixer(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dino_model_type: str,
        dino_model: nn.Module,
        stride: int,
        layer: int,
        thresh: float,
        build_basedata: bool,
        basedata_save_dir: str,
        basedata_ann_path: str,
        basedata_img_path: str,
        base_obj_cat: List[int],
        dino_min_size: int,
        dino_pixel_mean: Tuple[float],
        dino_pixel_std: Tuple[float],
        dino_pixel_norm: bool = True,
    ):
        super().__init__()
        # VLM R-CNN
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        # DINO
        self.base_obj_cat = base_obj_cat
        self.model_type = dino_model_type
        self.model = dino_model
        self.model = ParserNetMixer.patch_vit_resolution(self.model, stride=stride)

        self.p = self.model.patch_embed.patch_size
        self.stride = self.model.patch_embed.proj.stride
        self.layer = layer
        self.thresh = thresh
        self.build_basedata = build_basedata
        self.basedata_save_dir = basedata_save_dir
        self._feats = []
        self.hook_handlers = []
        self.num_patches = None
        self.load_size = None

        self.basedata_id2filename = {}
        self.basedata = []
        if not self.build_basedata:
            files = sorted(os.listdir(self.basedata_save_dir))
            for idx, file in enumerate(files):
                file_path = os.path.join(self.basedata_save_dir, file)
                data = torch.load(file_path, map_location='cpu')
                self.basedata.append(data)
                self.basedata_id2filename[idx] = file.split('.')[0]
            self.basedata = torch.stack(self.basedata).to('cuda')
            self.basedata_ann = COCO(basedata_ann_path)
            self.basedata_img_path = basedata_img_path

        self.dino_min_size = dino_min_size
        self.register_buffer("dino_pixel_mean", torch.tensor(dino_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("dino_pixel_std", torch.tensor(dino_pixel_std).view(-1, 1, 1), False)
        assert (
                self.dino_pixel_mean.shape == self.dino_pixel_std.shape
        ), f"{self.dino_pixel_mean} and {self.dino_pixel_std} have different shapes!"
        self.dino_pixel_norm = dino_pixel_norm

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        ret = {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

        dino_model_type = cfg.DINO.MODEL_TYPE
        ret.update({
            'dino_model_type': dino_model_type,
            'dino_model': ParserNetMixer.create_model(dino_model_type),
            'stride': cfg.DINO.STRIDE,
            'layer': cfg.DINO.LAYER,
            'thresh': cfg.DINO.THRESH,
            'build_basedata': cfg.DINO.BUILD_BASEDATA,
            'basedata_save_dir': cfg.DINO.BASEDATA_SAVE_DIR,
            'basedata_ann_path': cfg.DINO.BASEDATA_ANN_PATH,
            'basedata_img_path': cfg.DINO.BASEDATA_IMS_PATH,
            'base_obj_cat': cfg.DINO.BASE_OBJ_CAT,
            'dino_min_size': cfg.DINO.MIN_SIZE_TEST,
            'dino_pixel_mean': cfg.DINO.PIXEL_MEAN,
            'dino_pixel_std': cfg.DINO.PIXEL_STD,
            'dino_pixel_norm': cfg.DINO.PIXEL_NORM,
        })
        return ret

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        if 'dino' in model_type:
            model = torch.hub.load('facebookresearch/dino:main', model_type)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                'vit_small_patch16_224': 'dino_vits16',
                'vit_small_patch8_224': 'dino_vits8',
                'vit_base_patch16_224': 'dino_vitb16',
                'vit_base_patch8_224': 'dino_vitb8'
            }
            model = torch.hub.load('facebookresearch/dino:main', model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict['head.weight']
            del temp_state_dict['head.bias']
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(
            ParserNetMixer._fix_pos_enc(patch_size, stride), model)
        return model

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _register_hooks(self, layer: int, facets: List[str]) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx == layer:
                for facet in facets:
                    if facet == 'token':
                        self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                    elif facet == 'attn':
                        self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                    elif facet in ['key', 'query', 'value']:
                        self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                    else:
                        raise TypeError(f"{facet} is not a supported facet.")

    def _extract_features(self, batch: torch.Tensor, layers, facets) -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facets)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facets: List[str] = ['key'],
                            include_cls: bool = False) -> torch.Tensor:
        for facet in facets:
            assert facet in ['key', 'query', 'value', 'token', 'attn'], \
                f"""{facet} is not a supported facet for descriptors.
                choose from ['key' | 'query' | 'value' | 'token' | 'attn'] """
        self._extract_features(batch, layer, facets)

        descs = []
        for facet_idx, facet in enumerate(facets):
            x = self._feats[len(facets) - facet_idx - 1]
            if facet == 'attn':
                # assert self.model_type == "dino_vits8", \
                #     f"saliency maps are supported only for dino_vits model_type."
                head_idxs = [0, 2, 4, 5]
                cls_attn_map = x[:, head_idxs, 0, 1:].mean(dim=1)  # Bx(t-1)
                temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
                cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
                descs.append(cls_attn_maps)
                continue

            if facet == 'token':
                x.unsqueeze_(dim=1)  # Bx1xtxd

            if facet != 'token' or not include_cls:
                x = x[:, :, 1:, :]  # remove cls token

            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
            descs.append(desc)
        return descs

    def basedata_extract_save(self, batched_inputs, images):
        descriptors = self.extract_descriptors(
            images.tensor, self.layer, ["token"], include_cls=True)[0]
        save_dir = self.basedata_save_dir
        os.makedirs(save_dir, exist_ok=True)
        token = descriptors[0][0][0].cpu()
        image_name = batched_inputs[0]['file_name'].split('/')[-1]
        token_save_path = os.path.join(save_dir, image_name.split('.')[0] + '.pth')
        torch.save(token, token_save_path)
        return [{}]

    def find_nearest(self, images):
        tokens, descriptors, saliency_map = self.extract_descriptors(
            images.tensor, self.layer, ['token', 'key', 'attn'], include_cls=True)
        token = tokens[0][0][0]
        sim = nn.CosineSimilarity(dim=1, eps=1e-6)(self.basedata, token[None])
        nearest_idx = sim.argmax()
        base_filename = self.basedata_id2filename[nearest_idx.item()]
        return base_filename, descriptors, saliency_map

    def load_base_image(self, filename):
        base_image_path = self.basedata_img_path + filename + '.jpg'
        pil_image = Image.open(base_image_path).convert('RGB')
        pil_image = transforms.Resize(self.dino_min_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.dino_pixel_mean, std=self.dino_pixel_std)
        ])
        prep_img = prep(pil_image)[None, ...].to(self.device)

        return prep_img, pil_image


    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)
        assert 'ParserNet only support inference mode'


    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]],
                  do_postprocess: bool = True, ):
        assert len(batched_inputs) == 1, 'ParserNet only support batch size 1'

        if self.build_basedata:
            images = self.preprocess_dino_image(batched_inputs)
            return self.basedata_extract_save(batched_inputs, images)

        images = self.preprocess_image(batched_inputs)
        pos_category_ids = batched_inputs[0]['pos_category_ids'][0]
        if pos_category_ids in self.base_obj_cat:
            results = self.inference_base(batched_inputs, images)
        else:
            dino_images = self.preprocess_dino_image(batched_inputs)
            results = self.inference_novel(batched_inputs, images, dino_images)
            if self.vis_period == 1 or self.vis_period == 2:
                return results
                 
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return ParserNetMixer._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def inference_base(self, batched_inputs, images):
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features)
        results, _ = self.roi_heads(images, features, proposals)
        return results

    def inference_novel(self, batched_inputs, images, dino_images):
        base_filename, descriptors2, saliency_map2 = self.find_nearest(dino_images)
        num_patches2 = self.num_patches

        base_image_tensor, base_image = self.load_base_image(base_filename)
        descriptors1, saliency_map1 = self.extract_descriptors(
            base_image_tensor, self.layer, ['key', 'attn'])
        num_patches1 = self.num_patches

        # threshold saliency maps to get fg / bg masks
        fg_mask1 = saliency_map1[0] > self.thresh
        fg_mask2 = saliency_map2[0] > self.thresh

        # calculate similarity between image1 and image2 descriptors
        similarities = chunk_cosine_sim(descriptors1, descriptors2)

        # calculate correspondence
        sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
        sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
        sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
        sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]

        if self.vis_period == 1:
            return self.draw_correspondences(
                batched_inputs, base_image,
                nn_1, nn_2,
                fg_mask1, fg_mask2,
                num_patches1, num_patches2
            )

        key_image_id = base_filename.split('_')[0] + base_filename.split('_')[1]
        img = self.basedata_ann.loadImgs([int(key_image_id)])[0]
        anns = self.basedata_ann.imgToAnns[img['id']]
        height, width = img['height'], img['width']
        key_part_map = {}
        key_part_id_map = {}
        for instance_id, ann in enumerate(anns):
            category_id = ann['category_id']
            segm = ann["segmentation"]
            if isinstance(segm, list):  # polygon
                mask = polygons_to_bitmask(segm, height, width)
            elif isinstance(segm, dict):  # COCO RLE
                mask = mask_util.decode(segm)
            else:
                raise NotImplementedError
            mask_ = mask.astype(np.bool_)
            key_part_map[instance_id] = mask_
            key_part_id_map[instance_id] = category_id

        resize_key_part_map = {}
        for category_id in key_part_map:
            resize_part_map = cv2.resize(key_part_map[category_id].astype(np.float64),
                                         dsize=(num_patches1[1], num_patches1[0]))
            resize_key_part_map[category_id] = resize_part_map.astype(bool)

        bboxes, masks, category_ids = [], [], []
        dino_height, dino_width = dino_images.tensor.shape[-2:]
        for instance_id in resize_key_part_map:
            category_id = key_part_id_map[instance_id]

            mask1 = torch.tensor(resize_key_part_map[instance_id].reshape(-1), device=self.device)

            # indices_to_transfer = torch.nonzero(mask1, as_tuple=False).squeeze(dim=1)
            # indices_in2 = nn_1[indices_to_transfer]
            # mask2 = torch.zeros(num_patches2[0] * num_patches2[1],
            #                     dtype=torch.float32, device=self.device)
            # mask2[indices_in2] += 1
            # mask2[img2_indices_to_show] += 1
            # mask2 = mask2.reshape(num_patches2[0], num_patches2[1])
            # mask2 = (mask2 > 1.0).float()

            mask2 = mask1[nn_2]
            mask2 = (mask2 > 0.5).float() * fg_mask2.float()
            # mask2[img2_indices_to_show] += 1
            # mask2 = (mask2 > 1.0).float()
            mask2 = mask2.reshape(num_patches2[0], num_patches2[1])

            num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask2.cpu().numpy().astype(np.uint8), connectivity=8)
            if num_labels > 1:
                labels = torch.tensor(labels.astype(np.float), device=self.device)
                sub_masks = labels[None] == torch.arange(1, num_labels, device=self.device)[:,None,None]
                sub_areas = sub_masks.sum(-1).sum(-1)
                mask2 = (labels == (sub_areas.argmax() + 1)).float()

            resize_mask2 = F.interpolate(mask2[None][None], size=(dino_height, dino_width),
                                         mode='bilinear', align_corners=False)[0]
            resize_mask2 = (resize_mask2 > 0.0).float()
            bbox = mask_to_box(resize_mask2)[0]
            mask_ = resize_mask2[0]

            bboxes.append(bbox)
            masks.append(mask_)
            category_ids.append(category_id)

        if self.vis_period == 2:
            return self.draw_part_segmentation(
                batched_inputs,
                bboxes, masks, category_ids,
                base_filename, key_part_map,
            )

        height, width = batched_inputs[0]['image'].shape[-2:]
        if height > width:
            height, width = width, height
        scale = height / 800

        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features)
        results, _ = self.roi_heads.forward_top_proposals(images, features, proposals)

        pred_bboxes = results[0].pred_boxes.tensor * scale
        dino_bboxes = torch.stack(bboxes)
        iou_mat = pairwise_iou(Boxes(pred_bboxes), Boxes(dino_bboxes))

        max_val, max_idx = iou_mat.max(dim=0)
        keep = max_val > 0.3
        new_dino_bboxes = pred_bboxes[max_idx][keep]
        new_category_ids = torch.tensor(category_ids).to(self.device)[keep]

        if self.vis_period == 3:
            self.draw_part_segmentation_pred(
                batched_inputs, new_dino_bboxes, new_category_ids
            )

        res = Instances(dino_images.image_sizes[0])
        res.pred_boxes = Boxes(new_dino_bboxes)
        res.pred_classes = new_category_ids
        results = [res]

        return results

    def draw_correspondences(self, batched_inputs, key_image, nn_1, nn_2, fg_mask1, fg_mask2,
                             num_patches1, num_patches2, save_dir='output_corr',):

        image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)
        bbs_mask = nn_2[nn_1] == image_idxs

        # filter by saliency mask.
        fg_mask2_new_coors = nn_2[fg_mask2]
        fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=self.device)
        fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
        bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)
        indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)
        img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=self.device)[indices_to_show]
        img2_indices_to_show = nn_1[indices_to_show]

        image2 = Image.fromarray(batched_inputs[0]['image'].permute(1, 2, 0).numpy())
        image1 = key_image

        # coordinates in descriptor map's dimensions
        img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
        img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
        img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
        points1, points2 = [], []
        for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
            x1_show = (int(x1) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y1_show = (int(y1) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            x2_show = (int(x2) - 1) * self.stride[1] + self.stride[1] + self.p // 2
            y2_show = (int(y2) - 1) * self.stride[0] + self.stride[0] + self.p // 2
            points1.append((y1_show, x1_show))
            points2.append((y2_show, x2_show))

        assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
        num_points = len(points1)
        fig1, ax1 = plt.subplots()
        ax1.axis('off')
        fig2, ax2 = plt.subplots()
        ax2.axis('off')
        ax1.imshow(image1)
        ax2.imshow(image2)
        colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_points))
        radius1, radius2 = 8, 2
        for point1, point2, color in zip(points1, points2, colors):
            y1, x1 = point1
            circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, alpha=0.5, edgecolor='white', )
            circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, )
            ax1.add_patch(circ1_1)
            ax1.add_patch(circ1_2)

            y2, x2 = point2
            circ2_1 = plt.Circle((x2, y2), radius1, facecolor=color, alpha=0.5, edgecolor='white')
            circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, )
            ax2.add_patch(circ2_1)
            ax2.add_patch(circ2_2)

        os.makedirs(save_dir, exist_ok=True)
        image_name = batched_inputs[0]['file_name'].split('/')[-1]

        fig1_save_path = os.path.join(save_dir, image_name.split('.')[0] + '_key.JPEG')
        fig1.savefig(fig1_save_path, bbox_inches='tight', pad_inches=0)

        fig2_save_path = os.path.join(save_dir, image_name.split('.')[0] + '.JPEG')
        fig2.savefig(fig2_save_path, bbox_inches='tight', pad_inches=0)

        plt.close('all')
        return [{}]

    def draw_part_segmentation(self, batched_inputs, bboxes, masks, category_ids,
                               key_filename, key_part_map, save_dir='output_part_seg'):
        os.makedirs(save_dir, exist_ok=True)
        image_name = batched_inputs[0]['file_name'].split('/')[-1]

        base_image_path = self.basedata_img_path + key_filename + '.jpg'
        base_image = cv2.imread(base_image_path)

        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(key_part_map)))[:, :3]
        color_maps = [color * 255 for color in colors]
        # random.shuffle(color_maps)
        for instance_id in key_part_map:
            instance_color = color_maps[instance_id % len(color_maps)]
            mask_ = key_part_map[instance_id]
            for color_channel in range(3):
                base_image[mask_, color_channel] = \
                    base_image[mask_, color_channel] * 0.5 + \
                    instance_color[color_channel] * 0.5
        fig1_save_path = os.path.join(save_dir, image_name.split('.')[0] + '_key.jpg')
        cv2.imwrite(fig1_save_path, base_image)

        if self.input_format == 'RGB':
            image_show = batched_inputs[0]['image'][[2, 1, 0], :, :].permute(1, 2, 0).numpy()
        else:
            image_show = batched_inputs[0]['image'].permute(1, 2, 0).numpy()
        height, width = image_show.shape[:2]

        # from ...data.datasets.pascal_part import PASCAL_PART_BASE_CATEGORIES as CATEGORIES
        for instance_id, mask_ in enumerate(masks):
            mask = mask_[:height, :width].cpu().numpy().astype(np.bool_)
            instance_color = color_maps[instance_id % len(color_maps)]

            image_show = image_show.copy()
            bbox = bboxes[instance_id].cpu().numpy()
            # if bbox[0] < 50 or bbox[1] < 50:
            #     continue
            # image_show = cv2.rectangle(image_show,
            #     (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
            #     instance_color.tolist(), 2)

            for color_channel in range(3):
                image_show[mask, color_channel] = \
                    image_show[mask, color_channel] * 0.3 + \
                    instance_color[color_channel] * 0.7

            # category_id = category_ids[instance_id]
            # image_show = cv2.putText(image_show, "{}".format(CATEGORIES[category_id - 1]['name']),
            #                          (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_COMPLEX,
            #                          0.5, instance_color.tolist(), 1, cv2.LINE_AA)

        fig2_save_path = os.path.join(save_dir, image_name)
        cv2.imwrite(fig2_save_path, image_show)

        return [{}]

    def draw_part_segmentation_pred(self, batched_inputs, bboxes, category_ids,
                                    save_dir='output_part_seg_pred'):
        os.makedirs(save_dir, exist_ok=True)
        image_name = batched_inputs[0]['file_name'].split('/')[-1]

        colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(bboxes)))[:, :3]
        color_maps = [color * 255 for color in colors]
        # random.shuffle(color_maps)

        if self.input_format == 'RGB':
            image_show = batched_inputs[0]['image'][[2, 1, 0], :, :].permute(1, 2, 0).numpy()
        else:
            image_show = batched_inputs[0]['image'].permute(1, 2, 0).numpy()
        height, width = image_show.shape[:2]


        # from ...data.datasets.pascal_part import PASCAL_PART_BASE_CATEGORIES as CATEGORIES
        # for instance_id, mask_ in enumerate(masks):
        for instance_id, box in enumerate(bboxes):
            # mask = mask_[:height, :width].cpu().numpy().astype(np.bool_)
            instance_color = color_maps[instance_id % len(color_maps)]

            image_show = image_show.copy()
            bbox = bboxes[instance_id].cpu().numpy()
            # if bbox[0] < 50 or bbox[1] < 50:
            #     continue
            image_show = cv2.rectangle(image_show,
                (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                instance_color.tolist(), 2)

            # for color_channel in range(3):
            #     image_show[mask, color_channel] = \
            #         image_show[mask, color_channel] * 0.3 + \
            #         instance_color[color_channel] * 0.7

            # category_id = category_ids[instance_id]
            # image_show = cv2.putText(image_show, "{}".format(CATEGORIES[category_id - 1]['name']),
            #                          (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_COMPLEX,
            #                          0.5, instance_color.tolist(), 1, cv2.LINE_AA)

        fig2_save_path = os.path.join(save_dir, image_name)
        cv2.imwrite(fig2_save_path, image_show)

        return [{}]

    def preprocess_dino_image(self, batched_inputs):
        original_images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        if self.input_format == 'BGR':
            original_images = [ori_img[[2, 1, 0], :, :] for ori_img in original_images]
        if self.dino_pixel_norm:
            original_images = [x / 255.0 for x in original_images]
        images = [(x - self.dino_pixel_mean) / self.dino_pixel_std for x in original_images]
        images = ImageList.from_tensors(images,)
        return images

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        original_images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        resized_images = []
        for x in original_images:
            height, width = x.shape[-2:]
            swap = False
            if height > width:
                swap = True
                height, width = width, height
            scale = 800 / height
            new_height, new_width = height * scale, width * scale
            if swap:
                new_height, new_width = new_width, new_height
            resized_x = F.interpolate(x.float()[None], size=(int(new_height), int(new_width)), mode='bilinear', align_corners=False)[0]
            resized_images.append(resized_x.byte())

        images = [(x - self.pixel_mean) / self.pixel_std for x in resized_images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = custom_detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


def custom_detector_postprocess(
        results: Instances, output_height: int, output_width: int,
):
    """
    detector_postprocess with support on global_masks
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )

    resized_h, resized_w = results.image_size
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    return results


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()

    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y


def mask_to_box(masks):
    width_proj = masks.max(1)[0]
    height_proj = masks.max(2)[0]
    box_width, box_height = width_proj.sum(1), height_proj.sum(1)
    center_ws, _ = center_of_mass(width_proj[:, None, :])
    _, center_hs = center_of_mass(height_proj[:, :, None])
    boxes = torch.stack([center_ws - 0.5 * box_width, center_hs - 0.5 * box_height, center_ws + 0.5 * box_width,
                         center_hs + 0.5 * box_height], 1)
    return boxes
