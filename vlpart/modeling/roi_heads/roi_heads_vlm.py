# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import inspect
import logging

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou, pairwise_ioa
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.backbone.resnet import BottleneckBlock, ResNet
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.sampling import subsample_labels

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.roi_heads.roi_heads import Res5ROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads, _ScaleGradient

from .fast_rcnn_vlm import VLMFastRCNNOutputLayers


@ROI_HEADS_REGISTRY.register()
class VLMROIHeads(StandardROIHeads):
    @configurable
    def __init__(
        self,
        *,
        add_image_box: bool = False,
        image_box_size: float = 1.0,
        ws_num_props: int = 512,
        mult_object_score: bool = False,
        with_image_labels: bool = False,
        mask_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_image_box = add_image_box
        self.image_box_size = image_box_size
        self.ws_num_props = ws_num_props
        self.mult_object_score = mult_object_score
        self.with_image_labels = with_image_labels
        self.mask_weight = mask_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'image_box_size': cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE,
            'ws_num_props': cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS,
            'mult_object_score': cfg.MODEL.ROI_BOX_HEAD.MULT_OBJECT_SCORE,
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'mask_weight': cfg.MODEL.ROI_HEADS.MASK_WEIGHT,
        })
        return ret

    @classmethod
    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, height=pooler_resolution, width=pooler_resolution
        )
        box_head = build_box_head(cfg, pooled_shape)
        box_predictor = VLMFastRCNNOutputLayers(cfg, box_head.output_shape)

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(self, images, img_features, proposals, targets=None,
                ann_type='box', classifier_info=(None,None,None),
                dataset_source=None):
        del images
        features = [img_features[f] for f in self.box_in_features]
        image_sizes = [x.image_size for x in proposals]

        if self.training:
            if ann_type in ['box', 'part', 'ppart']:
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.get_top_proposals(proposals)

            pool_boxes = [x.proposal_boxes for x in proposals]
            pool_features = self.box_pooler(features, pool_boxes)

            box_features = self.box_head(pool_features)
            box_predictions = self.box_predictor(box_features, dataset_source=dataset_source)

            if ann_type in ['box', 'part']:
                loss = self.box_predictor.losses(
                    box_predictions, proposals, dataset_source=dataset_source)
            elif ann_type in ['ppart']:
                ###  ablation study on max_size or max_score
                # loss = self.box_predictor.part_classification_losses_x(
                #     box_predictions, proposals, targets)
                loss = self.box_predictor.part_classification_losses(
                    box_predictions, proposals, targets)
            else:
                loss = self.box_predictor.image_label_losses(
                    box_predictions, proposals, targets, classifier_info=classifier_info
                )

            if ann_type in ['box', 'part'] and targets[0].has('gt_masks'):
                mask_loss = self._forward_mask(img_features, proposals)
            else:
                mask_loss = self._get_empty_mask_loss(
                    img_features, proposals, device=proposals[0].objectness_logits.device)

            losses = {}
            losses.update(loss)
            losses.update({k: v * self.mask_weight for k, v in mask_loss.items()})

            return proposals, losses

        else:
            pool_boxes = [x.proposal_boxes for x in proposals]
            box_features = self.box_pooler(features, pool_boxes)
            box_features = self.box_head(box_features)

            box_predictions = self.box_predictor(box_features)
            boxes = self.box_predictor.predict_boxes(box_predictions, proposals)
            category_scores = self.box_predictor.predict_probs(box_predictions, proposals)

            if self.mult_object_score:
                if len(proposals) > 0 and proposals[0].has('scores'):
                    proposal_scores = [p.get('scores') for p in proposals]
                else:
                    proposal_scores = [p.get('objectness_logits').sigmoid() for p in proposals]
                scores = [(cs * ps[:, None]) ** 0.5 \
                          for cs, ps in zip(category_scores, proposal_scores)]
            else:
                scores = category_scores

            predictor = self.box_predictor
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            pred_instances = self.forward_with_given_boxes(img_features, pred_instances)

            return pred_instances, {}

    @torch.no_grad()
    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        for i, p in enumerate(proposals):
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
            if self.add_image_box:
                proposals[i] = self._add_image_box(p)
        return proposals

    @torch.no_grad()
    def _add_image_box(self, p):
        image_box = Instances(p.image_size)
        n = 1
        h, w = p.image_size
        f = self.image_box_size
        image_box.proposal_boxes = Boxes(
            p.proposal_boxes.tensor.new_tensor(
                [w * (1. - f) / 2.,
                    h * (1. - f) / 2.,
                    w * (1. - (1. - f) / 2.),
                    h * (1. - (1. - f) / 2.)]
                ).view(n, 4))
        image_box.objectness_logits = p.objectness_logits.new_ones(n)
        return Instances.cat([p, image_box])

    def _get_empty_mask_loss(self, features, proposals, device):
        return {'loss_mask': torch.zeros(
            (1,), device=device, dtype=torch.float32)[0]}

    def forward_top_proposals(self, images, img_features, proposals,):
        assert not self.training
        del images
        features = [img_features[f] for f in self.box_in_features]
        image_sizes = [x.image_size for x in proposals]

        proposals = self.get_top_proposals(proposals)
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = self.box_head(box_features)

        box_predictions = self.box_predictor(box_features)
        boxes = self.box_predictor.predict_boxes(box_predictions, proposals)
        scores = self.box_predictor.predict_probs(box_predictions, proposals)

        fake_scores = [scores_per_image[:,:2] for scores_per_image in scores]
        # if self.mult_object_score:
        #     if len(proposals) > 0 and proposals[0].has('scores'):
        #         proposal_scores = [p.get('scores') for p in proposals]
        #     else:
        #         proposal_scores = [p.get('objectness_logits').sigmoid() for p in proposals]
        #     scores = [(cs * ps[:, None]) ** 0.5 \
        #               for cs, ps in zip(category_scores, proposal_scores)]
        # else:
        #     scores = category_scores

        predictor = self.box_predictor
        pred_instances, _ = fast_rcnn_inference(
            boxes,
            fake_scores,
            image_sizes,
            # predictor.test_score_thresh,
            0.00,
            predictor.test_nms_thresh,
            predictor.test_topk_per_image,
        )

        return pred_instances, {}

@ROI_HEADS_REGISTRY.register()
class CascadeVLMROIHeads(CascadeROIHeads):
    @configurable
    def __init__(
        self,
        *,
        add_image_box: bool = False,
        image_box_size: float = 1.0,
        ws_num_props: int = 512,
        mult_proposal_score: bool = False,
        with_image_labels: bool = False,
        mask_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.add_image_box = add_image_box
        self.image_box_size = image_box_size
        self.ws_num_props = ws_num_props
        self.mult_proposal_score = mult_proposal_score
        self.with_image_labels = with_image_labels
        self.mask_weight = mask_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'image_box_size': cfg.MODEL.ROI_BOX_HEAD.IMAGE_BOX_SIZE,
            'ws_num_props': cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS,
            'mult_proposal_score': cfg.MODEL.ROI_BOX_HEAD.MULT_OBJECT_SCORE,
            'with_image_labels': cfg.WITH_IMAGE_LABELS,
            'mask_weight': cfg.MODEL.ROI_HEADS.MASK_WEIGHT,
        })
        return ret

    @classmethod
    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        assert len(cascade_bbox_reg_weights) == len(cascade_ious)
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, height=pooler_resolution, width=pooler_resolution
        )

        box_heads, box_predictors, proposal_matchers = [], [], []
        for match_iou, bbox_reg_weights in zip(cascade_ious, cascade_bbox_reg_weights):
            box_head = build_box_head(cfg, pooled_shape)
            box_heads.append(box_head)
            box_predictors.append(
                VLMFastRCNNOutputLayers(
                    cfg, box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                )
            )
            proposal_matchers.append(Matcher([match_iou], [0, 1], allow_low_quality_matches=False))
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_heads": box_heads,
            "box_predictors": box_predictors,
            "proposal_matchers": proposal_matchers,
        }


    def _forward_box(self, features, proposals, targets=None,
                ann_type='box', classifier_info=(None,None,None),
                dataset_source=None):
        if (not self.training) and self.mult_proposal_score:
            proposal_scores = [p.get('objectness_logits').sigmoid() for p in proposals]

        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes)
                if self.training and ann_type in ['box', 'part']:
                    proposals = self._match_and_label_boxes(
                        proposals, k, targets)
            predictions = self._run_stage(features, proposals, k,
                                          classifier_info=classifier_info,
                                          dataset_source=dataset_source)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(
                (predictions[0], predictions[1]), proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                if ann_type in ['box', 'part']:
                    loss = predictor.losses(
                        predictions, proposals, dataset_source=dataset_source)
                elif ann_type in ['ppart']:
                    loss = predictor.part_classification_losses_x(
                        predictions, proposals, targets)
                else:
                    loss = predictor.image_label_losses(
                        predictions, proposals, targets, classifier_info=classifier_info
                    )
                losses.update({k + "_stage{}".format(stage): v for k, v in loss.items()})
            return losses

        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 for s, ps in zip(scores, proposal_scores)]
            # if self.one_class_per_proposal:
            #     scores = [s * (s == s[:, :-1].max(dim=1)[0][:, None]).float() for s in scores]
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes((predictions[0], predictions[1]), proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances

    def forward(self, images, features, proposals, targets=None,
                ann_type='box', classifier_info=(None, None, None, None),
                dataset_source=None):
        del images
        if self.training:
            if ann_type in ['box', 'part']:
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.get_top_proposals(proposals)

            losses = self._forward_box(
                features, proposals, targets, ann_type=ann_type,
                classifier_info=classifier_info, dataset_source=dataset_source)

            if ann_type in ['box', 'part'] and targets[0].has('gt_masks'):
                mask_losses = self._forward_mask(features, proposals)
                losses.update({k: v * self.mask_weight for k, v in mask_losses.items()})
            else:
                losses.update(self._get_empty_mask_loss(
                    features, proposals, device=proposals[0].objectness_logits.device))
            return proposals, losses
        else:
            pred_instances = self._forward_box(
                features, proposals, classifier_info=classifier_info)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    @torch.no_grad()
    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        for i, p in enumerate(proposals):
            p.proposal_boxes.tensor = p.proposal_boxes.tensor.detach()
            if self.add_image_box:
                proposals[i] = self._add_image_box(p)
        return proposals

    @torch.no_grad()
    def _add_image_box(self, p):
        image_box = Instances(p.image_size)
        n = 1
        h, w = p.image_size
        f = self.image_box_size
        image_box.proposal_boxes = Boxes(
            p.proposal_boxes.tensor.new_tensor(
                [w * (1. - f) / 2.,
                    h * (1. - f) / 2.,
                    w * (1. - (1. - f) / 2.),
                    h * (1. - (1. - f) / 2.)]
                ).view(n, 4))
        image_box.objectness_logits = p.objectness_logits.new_ones(n)
        return Instances.cat([p, image_box])

    def _get_empty_mask_loss(self, features, proposals, device):
        return {'loss_mask': torch.zeros(
            (1,), device=device, dtype=torch.float32)[0]}

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals

    def _run_stage(self, features, proposals, stage,
                   classifier_info=(None,None,None),
                   dataset_source=None):
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        return self.box_predictor[stage](
            box_features,
            dataset_source=dataset_source,
            classifier_info=classifier_info)
