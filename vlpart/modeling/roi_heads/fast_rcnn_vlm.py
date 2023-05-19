# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import math
import copy
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit, giou_loss, smooth_l1_loss
import fvcore.nn.weight_init as weight_init

from detectron2.config import configurable
from detectron2.data.detection_utils import get_fed_loss_cls_weights
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.structures import Boxes, Instances, BitMasks, pairwise_iou, pairwise_ioa
from detectron2.utils.events import get_event_storage
import detectron2.utils.comm as comm
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

from ..utils.detic import load_class_freq, get_fed_loss_inds
from .zero_shot_classifier import ZeroShotClassifier, ZeroShotClassifierGroup


class VLMFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        zeroshot_cls=None,
        sync_caption_batch=False,
        use_sigmoid_ce=False,
        use_fed_loss=False,
        ignore_zero_cats=False,
        fed_loss_num_cat=50,
        dynamic_classifier=False,
        use_zeroshot_cls=False,
        use_zeroshot_cls_group=False,
        with_caption_loss=False,
        caption_loss_weight=1.0,
        neg_cap_weight=1.0,
        nouns_loss_weight=0.01,
        add_image_box=False,
        prior_prob=0.01,
        cat_freq_path='',
        fed_loss_freq_weight=0.5,
        part_loss_type='max_score',
        parsed_part_loss_weight=0.1,
        ignore_zero_cats_group=(False,),
        use_fed_loss_group=(False,),
        cat_freq_path_group=(''),
        **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            **kwargs,
        )
        self.sync_caption_batch = sync_caption_batch
        self.use_sigmoid_ce = use_sigmoid_ce
        self.use_fed_loss = use_fed_loss
        self.ignore_zero_cats = ignore_zero_cats
        self.fed_loss_num_cat = fed_loss_num_cat
        self.dynamic_classifier = dynamic_classifier
        self.use_zeroshot_cls = use_zeroshot_cls
        self.use_zeroshot_cls_group = use_zeroshot_cls_group
        self.with_caption_loss = with_caption_loss
        self.caption_loss_weight = caption_loss_weight
        self.neg_cap_weight = neg_cap_weight
        self.nouns_loss_weight = nouns_loss_weight
        self.add_image_box = add_image_box
        self.part_loss_type = part_loss_type
        self.parsed_part_loss_weight = parsed_part_loss_weight
        self.ignore_zero_cats_group = ignore_zero_cats_group
        self.use_fed_loss_group = use_fed_loss_group

        if self.use_zeroshot_cls_group:
            for idx, use_fed_loss in enumerate(use_fed_loss_group):
                if use_fed_loss:
                    freq_weight = load_class_freq(
                        cat_freq_path_group[idx], fed_loss_freq_weight)
                    self.register_buffer('freq_weight_{}'.format(idx), freq_weight)
                else:
                    self.register_buffer('freq_weight_{}'.format(idx), None)

        if self.use_fed_loss or self.ignore_zero_cats:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer('freq_weight', freq_weight)
        else:
            self.freq_weight = None

        if self.use_fed_loss and len(self.freq_weight) < self.num_classes:
            print('Extending federated loss weight')
            self.freq_weight = torch.cat(
                [self.freq_weight,
                 self.freq_weight.new_zeros(
                     self.num_classes - len(self.freq_weight))]
            )
        assert (not self.dynamic_classifier) or (not self.use_fed_loss)

        input_size = input_shape.channels * \
                     (input_shape.width or 1) * (input_shape.height or 1)

        # bbox_pred
        del self.bbox_pred
        self.bbox_pred = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, 4)
        )
        weight_init.c2_xavier_fill(self.bbox_pred[0])
        nn.init.normal_(self.bbox_pred[-1].weight, std=0.001)
        nn.init.constant_(self.bbox_pred[-1].bias, 0)

        # cls_score
        if self.use_zeroshot_cls_group:
            self.cls_score = zeroshot_cls
        elif self.use_zeroshot_cls:
            assert zeroshot_cls is not None
            self.cls_score = zeroshot_cls
        else:
            self.cls_score = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(inplace=True),
                nn.Linear(input_size, self.num_classes + 1),
            )
            weight_init.c2_xavier_fill(self.cls_score[0])
            nn.init.normal_(self.cls_score[-1].weight, mean=0, std=0.001)
            if self.use_sigmoid_ce:
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                nn.init.constant_(self.cls_score[-1].bias, bias_value)
            else:
                nn.init.constant_(self.cls_score[-1].bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'sync_caption_batch': cfg.MODEL.SYNC_CAPTION_BATCH,
            'use_sigmoid_ce': cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            'use_fed_loss': cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
            'ignore_zero_cats': cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS,
            'fed_loss_num_cat': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT,
            'dynamic_classifier': cfg.MODEL.DYNAMIC_CLASSIFIER,
            'use_zeroshot_cls': cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS,
            'use_zeroshot_cls_group': cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS_GROUP,
            'with_caption_loss': cfg.MODEL.ROI_BOX_HEAD.WITH_CAPTION_LOSS,
            'caption_loss_weight': cfg.MODEL.ROI_BOX_HEAD.CAPTION_LOSS_WEIGHT,
            'neg_cap_weight': cfg.MODEL.ROI_BOX_HEAD.NEG_CAP_WEIGHT,
            'nouns_loss_weight': cfg.MODEL.ROI_BOX_HEAD.NOUNS_LOSS_WEIGHT,
            'add_image_box': cfg.MODEL.ROI_BOX_HEAD.ADD_IMAGE_BOX,
            'prior_prob': cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB,
            'cat_freq_path': cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
            'fed_loss_freq_weight': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
            'parsed_part_loss_weight': cfg.MODEL.ROI_BOX_HEAD.PARSED_PART_LOSS_WEIGHT,
            'part_loss_type': cfg.MODEL.ROI_BOX_HEAD.PART_LOSS_TYPE,
            'ignore_zero_cats_group': cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS_GROUP,
            'use_fed_loss_group': cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS_GROUP,
            'cat_freq_path_group': cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH_GROUP,
        })
        if ret['use_zeroshot_cls_group']:
            ret['zeroshot_cls'] = ZeroShotClassifierGroup(cfg, input_shape)
        elif ret['use_zeroshot_cls']:
            ret['zeroshot_cls'] = ZeroShotClassifier(cfg, input_shape)

        return ret

    def forward(self, x, classifier_info=(None, None, None, None), dataset_source=None):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        cls_scores = []
        if classifier_info[0] is not None:
            cls_scores.append(self.cls_score(x, classifier=classifier_info[0]))
        else:
            if self.use_zeroshot_cls_group:
                cls_scores.append(self.cls_score(x, dataset_source=dataset_source))
            else:
                cls_scores.append(self.cls_score(x))

        if classifier_info[2] is not None:
            cap_cls = classifier_info[2]
            if self.sync_caption_batch:
                caption_scores = self.cls_score(x, classifier=cap_cls[:, :-1])
            else:
                caption_scores = self.cls_score(x, classifier=cap_cls)
            cls_scores.append(caption_scores)

        cls_scores = torch.cat(cls_scores, dim=1)  # B x C' or B x N or B x (C'+ N)
        proposal_deltas = self.bbox_pred(x)

        return cls_scores, proposal_deltas

    def predict_boxes(self, predictions, proposals):
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)

        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        cls_scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        cls_scores = cls_scores.split(num_inst_per_image, dim=0)

        final_scores = []
        for cls_score in cls_scores:
            final_score = cls_score.sigmoid() if self.use_sigmoid_ce else F.softmax(cls_score, dim=-1)
            final_scores.append(final_score)
        return final_scores

    def losses(self, predictions, proposals, dataset_source=None):
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        num_classes = scores.shape[1] - 1

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes, dataset_source)
        else:
            loss_cls = self.softmax_cross_entropy_loss(scores, gt_classes, dataset_source)

        loss_box_reg = self.box_reg_loss(proposal_boxes, gt_boxes, proposal_deltas, gt_classes,
                                         num_classes=num_classes)
        loss_dict = {
            "loss_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
            "loss_ppart": scores.new_zeros([1])[0],
            'loss_noun': scores.new_zeros([1])[0],
            # 'loss_caption': scores.new_zeros([1])[0],
        }
        return loss_dict

    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes, dataset_source=None):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]  # This is more robust than .sum() * 0.

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] - 1
        gt_classes = torch.clamp(gt_classes, max=C)  # multi-dataset train

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1  # B x (C + 1)
        target = target[:, :C]  # B x C

        weight = 1
        if self.use_zeroshot_cls_group:
            use_fed_loss = self.use_fed_loss_group[dataset_source]
            ignore_zero_cats = self.ignore_zero_cats_group[dataset_source]
            freq_weight = getattr(self, 'freq_weight_{}'.format(dataset_source))
        else:
            use_fed_loss = self.use_fed_loss
            ignore_zero_cats = self.ignore_zero_cats
            freq_weight = self.freq_weight

        if use_fed_loss and freq_weight is not None:  # fedloss
            appeared = get_fed_loss_inds(
                gt_classes,
                num_sample_cats=self.fed_loss_num_cat,
                C=C,
                weight=freq_weight)
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1  # C + 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.view(1, C).expand(B, C)
            weight = weight * fed_w.float()

        if ignore_zero_cats and freq_weight is not None:
            w = (freq_weight.view(-1) > 1e-4).float()
            weight = weight * w.view(1, C).expand(B, C)

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction='none')  # B x C
        loss = torch.sum(cls_loss * weight) / B
        return loss

    def softmax_cross_entropy_loss(self, pred_class_logits, gt_classes, dataset_source=None):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]
        C = pred_class_logits.shape[1] - 1
        gt_classes = torch.clamp(gt_classes, max=C)  # multi-dataset train
        loss = F.cross_entropy(
            pred_class_logits, gt_classes, reduction="mean")
        return loss

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, num_classes=-1):
        """
        Allow custom background index
        """
        num_classes = num_classes if num_classes > 0 else self.num_classes
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum")
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        return loss_box_reg / max(gt_classes.numel(), 1.0)

    def part_classification_losses(self, predictions, proposals, targets):
        scores, _ = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        assert self.use_sigmoid_ce
        loss_cls = self.sigmoid_cross_entropy_part_loss(scores, gt_classes)

        loss_dict = {
            "loss_cls": scores.new_zeros([1])[0],
            "loss_box_reg": scores.new_zeros([1])[0],
            "loss_ppart": loss_cls * self.parsed_part_loss_weight,
            'loss_noun': scores.new_zeros([1])[0],
        }
        return loss_dict

    def sigmoid_cross_entropy_part_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]  # This is more robust than .sum() * 0.

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] - 1
        gt_classes = torch.clamp(gt_classes, max=C)  # multi-dataset train

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1  # B x (C + 1)
        target = target[:, :C]  # B x C

        weight = 1
        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction='none')  # B x C
        loss = torch.sum(cls_loss * weight) / B
        return loss

    def part_classification_losses_x(self, predictions, proposals, targets):
        num_inst_per_image = [len(p) for p in proposals]
        cls_scores, _ = predictions
        cls_scores = cls_scores.split(num_inst_per_image, dim=0)  # B x n x (C + 1)

        B = len(cls_scores)
        cls_loss = cls_scores[0].new_zeros([1])[0]

        for idx, (cls_score, proposal, target) in enumerate(zip(cls_scores, proposals, targets)):
            if cls_score.shape[0] == 0:
                continue
            pos_category_ids = target.gt_classes
            for idx, label in enumerate(pos_category_ids):
                if self.part_loss_type == 'max_size':
                    loss_i = self._max_size_loss(cls_score, label, proposal)
                elif self.part_loss_type == 'max_score':
                    loss_i = self._max_score_loss(cls_score, label, proposal)
                else:
                    raise NotImplementedError

                cls_loss += loss_i / len(pos_category_ids)

        cls_loss = cls_loss / B
        loss_dict = {
            'loss_noun': cls_scores[0].new_zeros([1])[0],
            'loss_cls': cls_scores[0].new_zeros([1])[0],
            'loss_box_reg': cls_scores[0].new_zeros([1])[0],
            'loss_ppart': cls_loss * self.parsed_part_loss_weight,
        }

        return loss_dict


    def image_label_losses(self, predictions, proposals, targets, classifier_info=(None, None, None), ):
        num_inst_per_image = [len(p) for p in proposals]
        cls_scores, _ = predictions
        cls_scores = cls_scores.split(num_inst_per_image, dim=0)  # B x n x (C + 1)

        B = len(cls_scores)
        cls_loss = cls_scores[0].new_zeros([1])[0]

        for idx, (cls_score, proposal, target) in enumerate(zip(cls_scores, proposals, targets)):
            if cls_score.shape[0] == 0:
                continue

            pos_category_ids = target._pos_category_ids
            for label in pos_category_ids:
                loss_i = self._max_size_loss(cls_score, label, proposal)
                cls_loss += loss_i / len(pos_category_ids)

        cls_loss = cls_loss / B

        loss_dict = {
            'loss_noun': cls_loss * self.nouns_loss_weight,
            'loss_cls': cls_scores[0].new_zeros([1])[0],
            'loss_box_reg': cls_scores[0].new_zeros([1])[0],
            'loss_ppart': cls_scores[0].new_zeros([1])[0],
        }

        return loss_dict

    def _max_size_loss(self, score, label, p):
        target = score.new_zeros(score.shape[1])
        target[label] = 1.
        sizes = p.proposal_boxes.area()
        ind = sizes[:-1].argmax().item() if len(sizes) > 1 else 0
        loss = F.binary_cross_entropy_with_logits(
            score[ind], target, reduction='sum')
        return loss

    def _max_score_loss(self, score, label, p):
        target = score.new_zeros(score.shape[1])
        target[label] = 1.
        ind = score[:, label].argmax().item()
        loss = F.binary_cross_entropy_with_logits(
            score[ind], target, reduction='sum')
        return loss
