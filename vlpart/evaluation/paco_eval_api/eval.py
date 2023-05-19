# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
from collections import defaultdict, OrderedDict

import numpy as np

import pycocotools.mask as mask_utils

from lvis import LVISEval
from lvis.eval import Params

from .paco import PACO
from .results import PACOResults

SUPPORTED_ATTR_AP_TYPES = ["usual", "lb", "ub"]


def calculate_part_mask_area_outside_obj(dt, gt):
    overlap_precision = np.zeros((len(dt), len(gt)))
    for _i, _dt in enumerate(dt):
        for _j, _gt in enumerate(gt):
            overlap_precision[_i, _j] = _compute_mask_area_outside_impl(_dt, _gt)
    return overlap_precision


def calculate_part_box_area_outside_obj(dt, gt):
    overlap_precision = np.zeros((len(dt), len(gt)))
    for _i, _dt in enumerate(dt):
        for _j, _gt in enumerate(gt):
            overlap_precision[_i, _j] = _compute_box_area_outside_impl(_dt, _gt)
    return overlap_precision


def _compute_mask_area_outside_impl(mask_obj, mask_part):
    mask_obj = mask_utils.decode(mask_obj)
    mask_part = mask_utils.decode(mask_part)
    intersection = np.logical_and(mask_obj, mask_part).sum()
    area_part = mask_part.sum()
    return intersection.sum() / (area_part - intersection + 1e-7)


def _compute_box_area_outside_impl(box_obj, box_part):
    x1 = max(box_obj[0], box_part[0])
    y1 = max(box_obj[1], box_part[1])
    x2 = min(box_obj[2] + box_obj[0], box_part[2] + box_part[0])
    y2 = min(box_obj[3] + box_obj[1], box_part[3] + box_part[1])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_part = box_part[2] * box_part[3]
    return intersection / (area_part - intersection + 1e-7)


class PACOEval(LVISEval):
    def __init__(self, paco_gt, paco_dt, iou_type="segm", attr_ap_type="usual"):
        """Constructor for PACOEval.
        Args:
            paco_gt (PACO class instance, or str containing path of annotation file)
            paco_dt (PACOResult class instance, or str containing path of result file,
            or list of dict)
            iou_type (str): segm or bbox evaluation
            attr_ap_type (str): supports usual AP calc for attr +
                ub (upper bound) + lb (lower bound)
        """
        self.logger = logging.getLogger(__name__)

        if iou_type not in ["bbox", "segm"]:
            raise ValueError("iou_type: {} is not supported.".format(iou_type))

        if isinstance(paco_gt, PACO):
            self.lvis_gt = paco_gt
        elif isinstance(paco_gt, str):
            self.lvis_gt = PACO(paco_gt)
        else:
            raise TypeError("Unsupported type {} of paco_gt.".format(paco_gt))

        if isinstance(paco_dt, PACOResults):
            self.lvis_dt = paco_dt
        elif isinstance(paco_dt, (str, list)):
            self.lvis_dt = PACOResults(self.paco_gt, paco_dt)
        else:
            raise TypeError("Unsupported type {} of paco_dt.".format(paco_dt))

        # per-image per-category evaluation results
        self.eval_imgs = defaultdict(list)
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation

        self.params = Params(iou_type=iou_type)  # parameters
        self.results = OrderedDict()
        self.ious = {}  # ious between all gts and dts

        # map of im_id, attr_id to gt for attr eval
        self._gts_joint_attr = defaultdict(list)
        # map of im_id, attr_id to dt for attr eval
        self._dts_joint_attr = defaultdict(list)
        self.results_joint_attr = OrderedDict()

        self.params.img_ids = sorted(self.lvis_gt.get_img_ids())
        self.params.cat_ids = sorted(self.lvis_gt.get_cat_ids())
        self.params.joint_attr_ids = sorted(self.lvis_gt.get_joint_attr_ids())

        assert (
            attr_ap_type in SUPPORTED_ATTR_AP_TYPES
        ), f"unsupported attr_ap_type {attr_ap_type}"
        self.attr_ap_type = attr_ap_type

    def _prepare_joint_attr(self):
        """
        Prepare self._gts_joint_attr and self._dts_joint_attr for evaluation based on
        params.
        This function assumes that prepare_joint() has been run and self.dts is
        already populated
        """
        cat_ids = self.params.joint_attr_ids if self.params.joint_attr_ids else None
        self._gts_joint_attr = self.lvis_gt.im_joint_attr_cat_to_ann_id

        # For federated dataset evaluation we will filter out all dt for an
        # image which belong to categories not present in gt and not present in
        # the negative list for an image. In other words detector is not penalized
        # for categories about which we don't have gt information about their
        # presence or absence in an image.
        img_data = self.lvis_gt.load_imgs(ids=self.params.img_ids)
        # per image map of categories not present in image
        img_nl = {d["id"]: d["neg_category_ids_attrs"] for d in img_data}
        # per image list of categories present in image
        img_pl = defaultdict(set)
        for k in self._gts_joint_attr.keys():
            if self._gts_joint_attr[k]:
                img_pl[k[0]].add(k[1])

        # per image map of categoires which have missing gt. For these
        # categories we don't penalize the detector for false positives.
        self.img_nel_joint_attr = {
            d["id"]: d["not_exhaustive_category_ids_attrs"] for d in img_data
        }

        for img_id in self.params.img_ids:
            for cat_id in cat_ids:
                if cat_id not in img_nl[img_id] and cat_id not in img_pl[img_id]:
                    continue
                # detections for obj-attr are the same as detections for obj
                obj_cat = self.lvis_gt.joint_attr_cat_to_obj_attr[cat_id][0]
                self._dts_joint_attr[img_id, cat_id] = [
                    _ann["id"] for _ann in self._dts[(img_id, obj_cat)]
                ]

    def evaluate(self):
        """
        Run per image evaluation on given images and store results
        (a list of dict) in self.eval_imgs.
        """
        self.logger.info("Running per image object level evaluation.")
        self.logger.info("Evaluate annotation type *{}*".format(self.params.iou_type))

        self.params.img_ids = list(np.unique(self.params.img_ids))

        if self.params.use_cats:
            cat_ids = self.params.cat_ids
        else:
            cat_ids = [-1]

        self._prepare()

        self.ious = {
            (img_id, cat_id): self.compute_iou(img_id, cat_id)
            for img_id in self.params.img_ids
            for cat_id in cat_ids
        }

        # loop through images, area range, max detection number
        self.eval_imgs = [
            self.evaluate_img(img_id, cat_id, area_rng)
            for cat_id in cat_ids
            for area_rng in self.params.area_rng
            for img_id in self.params.img_ids
        ]

        # compute and save object and object-part level results
        # eval_imgs_ids is used to acess the precomputed object and object-part
        # level results for use in evaluating attrs
        self.eval_imgs_ids = {}
        idx = 0
        for cat_id in cat_ids:
            for area_rng in self.params.area_rng:
                for img_id in self.params.img_ids:
                    self.eval_imgs_ids[(cat_id, area_rng[0], area_rng[1], img_id)] = idx
                    idx += 1

    def evaluate_joint_attr(self):
        """
        Similar to self.evaluate(), this
        1. prepares the gt and dt for joint attr evaluation
        2. evalutes each image and joint attr category and stores results
        """
        self.logger.info("Running per image attr level evaluation.")
        self.logger.info("Evaluate annotation type *{}*".format(self.params.iou_type))

        if self.params.use_cats:
            joint_attr_ids = self.params.joint_attr_ids
        else:
            joint_attr_ids = [-1]

        self._prepare_joint_attr()

        self.eval_imgs_joint_attr = [
            self.evaluate_img_attr(img_id, cat_id, area_rng)
            for cat_id in joint_attr_ids
            for area_rng in self.params.area_rng
            for img_id in self.params.img_ids
        ]

    def _get_gt_dt_joint_attr(self, img_id, cat_id):
        """
        returns ann ids for gt and dt corressponding to cat_id in img_id
        """
        gt_id = self._gts_joint_attr[img_id, cat_id]
        dt_id = self._dts_joint_attr[img_id, cat_id]
        return gt_id, dt_id

    def compute_iou(self, img_id, cat_id):
        """
        Borrowed from https://github.com/lvis-dataset/lvis-api/blob/master/lvis/eval.py#L167
        This func also returns the ann ids of gt anf dt anns to be used for
        attr eval
        """
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return [], [], []

        # Sort detections in decreasing order of score.
        idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in idx]

        iscrowd = [int(False)] * len(gt)

        if self.params.iou_type == "segm":
            ann_type = "segmentation"
        elif self.params.iou_type == "bbox":
            ann_type = "bbox"
        else:
            raise ValueError("Unknown iou_type for iou computation.")
        gt_idx = [g["id"] for g in gt]
        dt_idx = [d["id"] for d in dt]
        gt = [g[ann_type] for g in gt]
        dt = [d[ann_type] for d in dt]

        # compute iou between each dt and gt region
        # will return array of shape len(dt), len(gt)
        ious = mask_utils.iou(dt, gt, iscrowd)
        return ious, gt_idx, dt_idx

    def compute_part_area_outside_object(self, dt, gt):
        """
        Computes the ratio of the area of part intersecting with object and
        area of the part outside the object
        """
        if len(gt) == 0 and len(dt) == 0:
            return [], [], []
        # Sort detections in decreasing order of score.
        idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in idx]
        if self.params.iou_type == "segm":
            ann_type = "segmentation"
        elif self.params.iou_type == "bbox":
            ann_type = "bbox"
        else:
            raise ValueError("Unknown iou_type for iou computation.")
        gt_idx = [g["id"] for g in gt]
        dt_idx = [d["id"] for d in dt]
        gt = [g[ann_type] for g in gt]
        dt = [d[ann_type] for d in dt]

        # compute iou between each dt and gt region
        # will return array of shape len(dt), len(gt)
        if self.params.iou_type == "segm":
            ious = calculate_part_mask_area_outside_obj(dt, gt)
        else:
            ious = calculate_part_box_area_outside_obj(dt, gt)

        return ious, gt_idx, dt_idx

    def evaluate_img(self, img_id, cat_id, area_rng):
        """
        Perform evaluation for single category and image.
        Borrowed from
        https://github.com/lvis-dataset/lvis-api/blob/master/lvis/eval.py#L193
        The difference compared to LVIS api is that we don't ignore fp
        predictions for object-parts when the image is NE for the object-part
        cat when the image is exhuastive for the object cat and the overlap of
        the predicted object-part is very low with all the object gt anns
        """
        gt, dt = self._get_gt_dt(img_id, cat_id)

        if len(gt) == 0 and len(dt) == 0:
            return None

        # Add another filed _ignore to only consider anns based on area range.
        for g in gt:
            if g["ignore"] or (g["area"] < area_rng[0] or g["area"] > area_rng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # Sort gt ignore last
        gt_idx = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gt_idx]

        # Sort dt highest score first
        dt_idx = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dt_idx]

        # load computed ious
        ious = (
            self.ious[img_id, cat_id][0][:, gt_idx]
            if len(self.ious[img_id, cat_id][0]) > 0
            else self.ious[img_id, cat_id]
        )

        num_thrs = len(self.params.iou_thrs)
        num_dt = len(dt)
        dt_m, gt_m, dt_ig, gt_ig = self._compute_matches(dt, gt, ious)

        # For LVIS we will ignore any unmatched detection if that category was
        # not exhaustively annotated in gt.
        ignore_ne_part = False
        if cat_id in self.lvis_gt.part_ids:
            obj_cat = self.lvis_gt.part_id_to_obj_id[cat_id]
            ignore_ne_part = (
                obj_cat not in self.img_nel[img_id] and cat_id in self.img_nel[img_id]
            )

        # if cat_id in self.lvis_gt.obj_ids or not ignore_ne_part:
        dt_ig_mask = [
            d["area"] < area_rng[0]
            or d["area"] > area_rng[1]
            or d["category_id"] in self.img_nel[d["image_id"]]
            for d in dt
        ]
        dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
        dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt
        # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
        dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == 0, dt_ig_mask))

        # begin change from original lvis code
        if ignore_ne_part:
            # above, we ignore all fp for NE category
            # if NE category is an obj-part cat, and the obj cat is
            # exhaustively annotate, ignore an fp only when it has a large
            # enough overlap with a gt object cat ann

            gt_for_obj_cat, _ = self._get_gt_dt(img_id, obj_cat)
            # gt for obj can only be 0, if the gt for obj-part is also 0.
            # in this case this img cannot be NE, but will be N
            if len(gt_for_obj_cat) and len(dt):
                overlap_precision, _, _ = self.compute_part_area_outside_object(
                    dt, gt_for_obj_cat
                )
                for _i, _d in enumerate(dt):
                    for iou_thresh_idx in range(num_thrs):
                        # if dt unmatched
                        if dt_m[iou_thresh_idx, _i] == 0.0:
                            if _d["area"] < area_rng[0] or _d["area"] > area_rng[1]:
                                # continue to ignore if area mismatch
                                continue
                            # when obj-part dt has a reasonable match with some
                            # gt obj, ignore it, else don't
                            if np.nonzero(overlap_precision[_i])[0].size > 0:
                                dt_overlap = overlap_precision[_i]
                                if (
                                    np.max(dt_overlap[np.nonzero(dt_overlap)[0]])
                                    >= self.params.iou_thrs[iou_thresh_idx]
                                ):
                                    dt_ig[iou_thresh_idx, _i] = False
                                else:
                                    dt_ig[iou_thresh_idx, _i] = True
                            else:
                                dt_ig[iou_thresh_idx, _i] = False
        # end change from original lvis code
        return {
            "image_id": img_id,
            "category_id": cat_id,
            "area_rng": area_rng,
            "dt_ids": [d["id"] for d in dt],
            "gt_ids": [g["id"] for g in gt],
            "dt_matches": dt_m,
            "gt_matches": gt_m,
            "dt_scores": [d["score"] for d in dt],
            "gt_ignore": gt_ig,
            "dt_ignore": dt_ig,
        }

    def _compute_matches(
        self, dt, gt, ious, dt_idx_cat=None, gt_idx_cat=None, dt_idx=None, gt_idx=None
    ):
        num_thrs = len(self.params.iou_thrs)
        num_gt = len(gt)
        num_dt = len(dt)

        # Array to store the "id" of the matched dt/gt
        gt_m = np.zeros((num_thrs, num_gt))
        dt_m = np.zeros((num_thrs, num_dt))

        gt_ig = np.array([g["_ignore"] for g in gt])
        dt_ig = np.zeros((num_thrs, num_dt))

        if dt_idx_cat is not None:
            assert (
                gt_idx_cat is not None
            ), "both gt and dt indexes should be available for attr match computation"
            if len(dt_idx_cat) == 0 or len(gt_idx_cat) == 0 or len(ious) == 0:
                return dt_m, gt_m, dt_ig, gt_ig

            # extract based on object level ids
            ious = ious[:, gt_idx_cat]
            ious = ious[dt_idx_cat, :]
            # re-org based on new sorted ids
            ious = ious[:, np.array(gt_idx)]
            ious = ious[np.array(dt_idx), :]

        for iou_thr_idx, iou_thr in enumerate(self.params.iou_thrs):

            for dt_idx, _dt in enumerate(dt):
                iou = min([iou_thr, 1 - 1e-10])
                # information about best match so far (m=-1 -> unmatched)
                # store the gt_idx which matched for _dt
                m = -1
                for gt_idx, _ in enumerate(gt):
                    # if this gt already matched continue
                    if gt_m[iou_thr_idx, gt_idx] > 0:
                        continue
                    # if _dt matched to reg gt, and on ignore gt, stop
                    if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                        break
                    # continue to next gt unless better match made
                    if ious[dt_idx, gt_idx] < iou:
                        continue
                    # if match successful and best so far, store appropriastely
                    iou = ious[dt_idx, gt_idx]
                    m = gt_idx

                # No match found for _dt, go to next _dt
                if m == -1:
                    continue

                # if gt to ignore for some reason update dt_ig.
                # Should not be used in evaluation.
                dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
                # _dt match found, update gt_m, and dt_m with "id"
                dt_m[iou_thr_idx, dt_idx] = gt[m]["id"]
                gt_m[iou_thr_idx, m] = _dt["id"]

        return dt_m, gt_m, dt_ig, gt_ig

    def evaluate_img_attr(self, img_id, cat_id, area_rng):
        """
        Perform evaluation for single joint attr category and image.
        It uses stored ious for the object level detections, hence assumes
        that evaluate_img() has been run already
        """
        obj_cat = self.lvis_gt.joint_attr_cat_to_obj_attr[cat_id][0]
        attr_id = self.lvis_gt.joint_attr_cat_to_obj_attr[cat_id][1]

        # load ious corresspoding to object level dt and gt
        # orig_gt_ids and orig_dt_ids are ann ids
        ious, orig_gt_ids, orig_dt_ids = (
            self.ious[img_id, obj_cat]
            if len(self.ious[img_id, obj_cat][0]) > 0
            else self.ious[img_id, obj_cat]
        )

        # load obj level data for gt and dt
        gt_obj, dt_obj = self._get_gt_dt(img_id, obj_cat)
        # used later to check if match existed at obj level
        gt_id_to_gt = {g["id"]: g for g in gt_obj}

        if len(gt_obj) == 0 and len(dt_obj) == 0:
            return None

        # load ann ids which corresspond to the joint attr cat
        gt_id, dt_id = self._get_gt_dt_joint_attr(img_id, cat_id)
        if len(gt_id) == 0 and len(dt_id) == 0:
            return None

        # find indices in object level gt and dt to use for attr eval
        gt_idx_cat = np.array([_i for _i, x in enumerate(orig_gt_ids) if x in gt_id])
        # orig_dt_ids dt ann ids sorted by obj score; these corresspond to the
        # ordering for dt in ious
        dt_idx_cat = np.array([_i for _i, x in enumerate(orig_dt_ids) if x in dt_id])

        # gt and dt for attr eval
        gt = [x for x in gt_obj if x["id"] in gt_id]
        # NOTE: all the detections are used since it is assumed that the all
        # attributes are predicted with some score for all object and
        # object-part detections
        dt = [x for x in dt_obj if x["id"] in dt_id]

        # Add another filed _ignore to only consider anns based on area range.
        for g in gt:
            if g["ignore"] or (g["area"] < area_rng[0] or g["area"] > area_rng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # Sort gt ignore last
        gt_idx = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gt_idx]

        # Sort dt highest score first
        if self.attr_ap_type == "usual":
            dt_idx = np.argsort(
                [-d["score"] * d["attribute_probs"][attr_id] for d in dt],
                kind="mergesort",
            )
        else:
            dt_idx = np.argsort([-d["score"] for d in dt], kind="mergesort")

        dt = [dt[i] for i in dt_idx]

        num_thrs = len(self.params.iou_thrs)
        num_dt = len(dt)
        dt_m, gt_m, dt_ig, gt_ig = self._compute_matches(
            dt, gt, ious, dt_idx_cat, gt_idx_cat, dt_idx, gt_idx
        )

        # For LVIS we will ignore any unmatched detection if that category was
        # not exhaustively annotated in gt.
        # we will ignore any unmatched detection if that dt had a match with a gt
        # using only bbox matching, and gt has unknown for that attr type.
        # This means this dt matched with a gt box without annotations

        # for the tps at this point:
        #     don't touch them
        # for the fps (i.e. dt with no match):
        #     if outside area range:
        #         ignore
        #     if at object level, match with a gt box with annotations:
        #         don't ignore
        #     if at object level, match with a gt box with no attr annotations:
        #         ignore
        #     if matches with no box at obj level and set to ignore:
        #         ignore if the det is ignored at object level (
        #           i.e. ne obj or is a part with a high overlap with a gt with
        #           no part anns
        #         ) else dont
        attr_type = self.lvis_gt.attr_idxs_to_type[attr_id]
        dt_ig_mask = [d["area"] < area_rng[0] or d["area"] > area_rng[1] for d in dt]
        dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
        dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt
        dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == 0, dt_ig_mask))

        # look up matches based on object level prediction for these attr
        eval_id = self.eval_imgs_ids[(obj_cat, area_rng[0], area_rng[1], img_id)]
        dt_ids_for_obj = {
            dt_id: _i for _i, dt_id in enumerate(self.eval_imgs[eval_id]["dt_ids"])
        }
        dt_ids_for_attr = {_i: d["id"] for _i, d in enumerate(dt)}
        dt_ids_for_attr_in_obj = [
            dt_ids_for_obj[dt_ids_for_attr[_i]] for _i in range(len(dt_ids_for_attr))
        ]
        matches_for_obj = self.eval_imgs[eval_id]["dt_matches"][
            :, dt_ids_for_attr_in_obj
        ]
        dt_ignores_for_obj = self.eval_imgs[eval_id]["dt_ignore"][
            :, dt_ids_for_attr_in_obj
        ]
        assert [d["id"] for d in dt] == [
            self.eval_imgs[eval_id]["dt_ids"][_i] for _i in dt_ids_for_attr_in_obj
        ]

        for _i, _d in enumerate(dt):
            for iou_thresh_idx in range(num_thrs):
                # if dt unmatched
                if dt_m[iou_thresh_idx, _i] == 0.0:
                    if dt_ig[iou_thresh_idx, _i]:
                        # dont penalize if area mismatch
                        continue
                    gt_match = matches_for_obj[iou_thresh_idx, _i]
                    if gt_match > 0.0:
                        if gt_id_to_gt[int(gt_match)][f"unknown_{attr_type}"]:
                            dt_ig[iou_thresh_idx, _i] = True
                        else:
                            dt_ig[iou_thresh_idx, _i] = False
                    else:
                        dt_ig[iou_thresh_idx, _i] = dt_ignores_for_obj[
                            iou_thresh_idx, _i
                        ]

        if self.attr_ap_type == "ub":
            # if a dt has a match at some thresh, then set it's score to 1.0,
            # else set score to 0.0
            scores = []
            for _i, _d in enumerate(dt):
                if dt_m[0, _i] > 0.0:
                    # tp
                    scores.append(1.0)
                else:
                    # fp
                    scores.append(0.0)
        elif self.attr_ap_type == "lb":
            scores = [d["score"] for d in dt]
        else:
            scores = [d["score"] * d["attribute_probs"][attr_id] for d in dt]
        # store results for given image and category
        return {
            "image_id": img_id,
            "category_id": cat_id,
            "obj_cat": obj_cat,
            "attr_cat": attr_id,
            "area_rng": area_rng,
            "dt_ids": [d["id"] for d in dt],
            "gt_ids": [g["id"] for g in gt],
            "dt_matches": dt_m,
            "gt_matches": gt_m,
            "dt_scores": scores,
            "gt_ignore": gt_ig,
            "dt_ignore": dt_ig,
        }

    def accumulate_joint_attr(self):
        """
        Borrowed from https://github.com/lvis-dataset/lvis-api/blob/master/lvis/eval.py#L293
        Change: runs on joint_attr_ids and eval_imgs_joint_attr
        """
        self.logger.info("Accumulating evaluation results.")

        if not self.eval_imgs_joint_attr:
            self.logger.warn("Please run evaluate first.")

        if self.params.use_cats:
            cat_ids = self.params.joint_attr_ids
        else:
            cat_ids = [-1]

        num_thrs = len(self.params.iou_thrs)
        num_recalls = len(self.params.rec_thrs)
        num_cats = len(cat_ids)
        num_area_rngs = len(self.params.area_rng)
        num_imgs = len(self.params.img_ids)

        # -1 for absent categories
        precision = -np.ones((num_thrs, num_recalls, num_cats, num_area_rngs))
        recall = -np.ones((num_thrs, num_cats, num_area_rngs))

        # Initialize dt_pointers
        dt_pointers = {}
        for cat_idx in range(num_cats):
            dt_pointers[cat_idx] = {}
            for area_idx in range(num_area_rngs):
                dt_pointers[cat_idx][area_idx] = {}

        # Per category evaluation
        for cat_idx in range(num_cats):
            Nk = cat_idx * num_area_rngs * num_imgs
            for area_idx in range(num_area_rngs):
                Na = area_idx * num_imgs
                E = [
                    self.eval_imgs_joint_attr[Nk + Na + img_idx]
                    for img_idx in range(num_imgs)
                ]
                # Remove elements which are None
                E = [e for e in E if e is not None]
                if len(E) == 0:
                    continue

                # Append all scores: shape (N,)
                dt_scores = np.concatenate([e["dt_scores"] for e in E], axis=0)
                dt_ids = np.concatenate([e["dt_ids"] for e in E], axis=0)

                dt_idx = np.argsort(-dt_scores, kind="mergesort")
                dt_scores = dt_scores[dt_idx]
                dt_ids = dt_ids[dt_idx]

                dt_m = np.concatenate([e["dt_matches"] for e in E], axis=1)[:, dt_idx]
                dt_ig = np.concatenate([e["dt_ignore"] for e in E], axis=1)[:, dt_idx]

                gt_ig = np.concatenate([e["gt_ignore"] for e in E])
                # num gt anns to consider
                num_gt = np.count_nonzero(gt_ig == 0)

                if num_gt == 0:
                    continue

                tps = np.logical_and(dt_m, np.logical_not(dt_ig))
                fps = np.logical_and(np.logical_not(dt_m), np.logical_not(dt_ig))

                tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                dt_pointers[cat_idx][area_idx] = {
                    "dt_ids": dt_ids,
                    "tps": tps,
                    "fps": fps,
                }

                for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    tp = np.array(tp)
                    fp = np.array(fp)
                    num_tp = len(tp)
                    rc = tp / num_gt

                    if num_tp:
                        recall[iou_thr_idx, cat_idx, area_idx] = rc[-1]
                    else:
                        recall[iou_thr_idx, cat_idx, area_idx] = 0

                    # np.spacing(1) ~= eps
                    pr = tp / (fp + tp + np.spacing(1))
                    pr = pr.tolist()
                    # if iou_thr_idx == 0 and area_idx == 0:
                    #     import pdb; pdb.set_trace()

                    # Replace each precision value with the maximum precision
                    # value to the right of that recall level. This ensures
                    # that the  calculated AP value will be less suspectable
                    # to small variations in the ranking.
                    for i in range(num_tp - 1, 0, -1):
                        if pr[i] > pr[i - 1]:
                            pr[i - 1] = pr[i]

                    rec_thrs_insert_idx = np.searchsorted(
                        rc, self.params.rec_thrs, side="left"
                    )

                    pr_at_recall = [0.0] * num_recalls

                    try:
                        for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                            pr_at_recall[_idx] = pr[pr_idx]
                    except:
                        pass
                    precision[iou_thr_idx, :, cat_idx, area_idx] = np.array(
                        pr_at_recall
                    )

        self.eval_joint_attr = {
            "params": self.params,
            "counts": [num_thrs, num_recalls, num_cats, num_area_rngs],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "dt_pointers": dt_pointers,
        }

    def _summarize_joint_attr(
        self, summary_type, iou_thr=None, area_rng="all", freq_group_idx=None
    ):
        aidx = [
            idx
            for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]

        if summary_type == "ap":
            s = self.eval_joint_attr["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                s = s[:, :, self.freq_groups[freq_group_idx], aidx]
            else:
                s = s[:, :, :, aidx]
        else:
            s = self.eval_joint_attr["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, :, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize_joint_attr(self):
        """Compute and display summary metrics for joint evaluation results."""
        if not self.eval_joint_attr:
            raise RuntimeError("Please run accumulate() first.")

        max_dets = self.params.max_dets

        self.results_joint_attr["AP"] = self._summarize_joint_attr("ap")
        self.results_joint_attr["AP50"] = self._summarize_joint_attr("ap", iou_thr=0.50)
        self.results_joint_attr["AP75"] = self._summarize_joint_attr("ap", iou_thr=0.75)
        self.results_joint_attr["APs"] = self._summarize_joint_attr(
            "ap", area_rng="small"
        )
        self.results_joint_attr["APm"] = self._summarize_joint_attr(
            "ap", area_rng="medium"
        )
        self.results_joint_attr["APl"] = self._summarize_joint_attr(
            "ap", area_rng="large"
        )

        key = "AR@{}".format(max_dets)
        self.results_joint_attr[key] = self._summarize_joint_attr("ar")

        for area_rng in ["small", "medium", "large"]:
            key = "AR{}@{}".format(area_rng[0], max_dets)
            self.results_joint_attr[key] = self._summarize_joint_attr(
                "ar", area_rng=area_rng
            )

    def run(self):
        """Wrapper function which calculates the results. Overrides LVIS run()"""
        self.evaluate()
        self.evaluate_joint_attr()
        self.accumulate()
        self.accumulate_joint_attr()
        self.summarize()
        self.summarize_joint_attr()

    def print_results_joint_attr(self):
        """
        Similar to print_results() in LVIS eval; supports attributes
        """
        template = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} catIds={:>3s}] = {:0.3f}"
        for key, value in self.results_joint_attr.items():
            max_dets = self.params.max_dets
            if "AP" in key:
                title = "Average Precision"
                _type = "(AP)"
            else:
                title = "Average Recall"
                _type = "(AR)"

            if len(key) > 2 and key[2].isdigit():
                iou_thr = float(key[2:]) / 100
                iou = "{:0.2f}".format(iou_thr)
            else:
                iou = "{:0.2f}:{:0.2f}".format(
                    self.params.iou_thrs[0], self.params.iou_thrs[-1]
                )

            if len(key) > 2 and key[2] in ["r", "c", "f"]:
                cat_group_name = key[2]
            else:
                cat_group_name = "all"

            if len(key) > 2 and key[2] in ["s", "m", "l"]:
                area_rng = key[2]
            else:
                area_rng = "all"

            print(
                template.format(
                    title, _type, iou, area_rng, max_dets, cat_group_name, value
                )
            )
