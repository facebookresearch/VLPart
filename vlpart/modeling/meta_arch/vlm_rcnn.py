# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
import itertools
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like, batched_nms
from detectron2.structures import ImageList, Boxes, Instances, BitMasks, ROIMasks
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

import clip


@META_ARCH_REGISTRY.register()
class VLMRCNN(nn.Module):
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
        eval_proposal=False,
        with_image_labels=False,
        fp16=False,
        sync_caption_batch=False,
        roi_head_name='',
        cap_batch_ratio=4,
        with_caption=False,
        text_encoder_type="ViT-B/32",
        text_encoder_dim=512,
        dynamic_classifier=False,
    ):
        super().__init__()

        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.eval_proposal = eval_proposal
        self.with_image_labels = with_image_labels
        self.fp16 = fp16
        self.with_caption = with_caption
        self.sync_caption_batch = sync_caption_batch
        self.roi_head_name = roi_head_name
        self.cap_batch_ratio = cap_batch_ratio
        self.text_encoder_dim = text_encoder_dim

        self.dynamic_classifier = dynamic_classifier
        if self.dynamic_classifier:
            self.freq_weight = kwargs.pop('freq_weight')
            self.num_classes = kwargs.pop('num_classes')
            self.num_sample_cats = kwargs.pop('num_sample_cats')

        if self.with_caption:
            assert not self.dynamic_classifier
            self.text_encoder = build_text_encoder(
                pretrain=True, visual_type=text_encoder_type)
            for v in self.text_encoder.parameters():
                v.requires_grad = False

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

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

        ret.update({
            'eval_proposal': cfg.MODEL.EVAL_PROPOSAL,
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'fp16': cfg.FP16,
            'with_caption': cfg.MODEL.WITH_CAPTION,
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'roi_head_name': cfg.MODEL.ROI_HEADS.NAME,
            'cap_batch_ratio': cfg.MODEL.CAP_BATCH_RATIO,
            'text_encoder_type': cfg.MODEL.TEXT_ENCODER_TYPE,
            'text_encoder_dim': cfg.MODEL.TEXT_ENCODER_DIM,
        })

        if ret['dynamic_classifier']:
            ret['freq_weight'] = load_class_freq(
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
                cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT)
            ret['num_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
            ret['num_sample_cats'] = cfg.MODEL.NUM_SAMPLE_CATS

        return ret

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        """
        if not self.training:
            return self.inference(batched_inputs)

        # annotation
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        for inst, x in zip(gt_instances, batched_inputs):
            inst._ann_type = x['ann_type'] if 'ann_type' in x else 'box'
            if 'pos_category_ids' in x and len(x['pos_category_ids']) > 0:
                inst._pos_category_ids = x['pos_category_ids']
            else:
                inst._pos_category_ids = inst.gt_classes.unique()
            inst._dataset_source = x['dataset_source'] if 'dataset_source' in x else 0
        ann_types = [inst._ann_type for inst in gt_instances]
        assert len(set(ann_types)) == 1
        ann_type = ann_types[0]

        dataset_sources = [inst._dataset_source for inst in gt_instances]
        assert len(set(dataset_sources)) == 1
        dataset_source = dataset_sources[0]

        # image
        images = self.preprocess_image(batched_inputs)

        # backbone
        if self.fp16:
            with autocast():
                features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)

        # region proposal
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)

        # roi head
        proposals, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances,
            ann_type=ann_type,
            dataset_source=dataset_source,
        )

        # record losses
        losses = {}
        losses.update(detector_losses)
        if ann_type in ['box', 'part']:
            losses.update(proposal_losses)
        else:  # ignore proposal loss for non-bbox data
            losses.update({k: v * 0 for k, v in proposal_losses.items()})

        return losses


    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features)
        if self.eval_proposal:
            results = self.proposals_to_instances(proposals)
        else:
            results, _ = self.roi_heads(images, features, proposals)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            max_shape = images.tensor.shape[2:]
            return VLMRCNN._postprocess(results, batched_inputs, images.image_sizes, max_shape)
        else:
            return results

    def proposals_to_instances(self, proposals, score_thresh=0.001, nms_thresh=0.7):
        results = []
        for proposals_per_img in proposals:
            image_shape = proposals_per_img.image_size
            scores = proposals_per_img.get('objectness_logits').sigmoid()
            boxes = proposals_per_img.get('proposal_boxes').tensor

            scores = scores.reshape(-1)
            boxes = Boxes(boxes.reshape(-1, 4))
            boxes.clip(image_shape)
            boxes = boxes.tensor.view(-1, 4)

            filter_mask = scores > score_thresh
            boxes = boxes[filter_mask]
            scores = scores[filter_mask]

            keep = batched_nms(boxes, scores, torch.zeros_like(scores), nms_thresh)
            boxes, scores = boxes[keep], scores[keep]

            result = Instances(image_shape)
            result.pred_boxes = Boxes(boxes)
            result.scores = scores
            results.append(result)

        return results


    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        original_images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in original_images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes, max_shape):
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
            r = custom_detector_postprocess(results_per_image, height, width, max_shape)
            processed_results.append({"instances": r})
        return processed_results


def custom_detector_postprocess(
        results: Instances, output_height: int, output_width: int,
        max_shape, mask_threshold: float = 0.5
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

    if results.has("pred_global_masks"):
        mask_pred_per_image = F.interpolate(results.pred_global_masks.unsqueeze(1), size=max_shape, mode="bilinear",
                                            align_corners=False)
        mask_pred_per_image = mask_pred_per_image[:, :, :resized_h, :resized_w]
        mask_pred_per_image = F.interpolate(mask_pred_per_image, size=new_size, mode='bilinear',
                                            align_corners=False).squeeze(1)
        results.pred_masks = mask_pred_per_image > mask_threshold

    elif results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    return results
