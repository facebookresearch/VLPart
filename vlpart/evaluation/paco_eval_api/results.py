# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from copy import deepcopy

import pycocotools.mask as mask_utils

from .paco import PACO


class PACOResults(PACO):
    def __init__(self, paco_gt, results, max_dets=300):
        """Constructor for PACO results.
        Args:
            paco_gt (PACO class instance, or str containing path of
            annotation file)
            results (str containing path of result file or a list of dicts)
            max_dets (int):  max number of detections per image. The official
            value of max_dets for PACO is 300.
        """
        if isinstance(paco_gt, PACO):
            self.dataset = deepcopy(paco_gt.dataset)
        elif isinstance(paco_gt, str):
            self.dataset = self._load_json(paco_gt)
        else:
            raise TypeError("Unsupported type {} of paco_gt.".format(paco_gt))

        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading and preparing results.")

        if isinstance(results, str):
            result_anns = self._load_json(results)
        else:
            # this path way is provided to avoid saving and loading result
            # during training.
            self.logger.warn("Assuming user provided the results in correct format.")
            result_anns = results

        assert isinstance(result_anns, list), "results is not a list."

        if max_dets >= 0:
            result_anns = self.limit_dets_per_image(result_anns, max_dets)

        if "bbox" in result_anns[0]:
            for id, ann in enumerate(result_anns):
                x1, y1, w, h = ann["bbox"]
                x2 = x1 + w
                y2 = y1 + h

                if "segmentation" not in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]

                ann["area"] = w * h
                ann["id"] = id + 1

        elif "segmentation" in result_anns[0]:
            for id, ann in enumerate(result_anns):
                # Only support compressed RLE format as segmentation results
                ann["area"] = mask_utils.area(ann["segmentation"])

                if "bbox" not in ann:
                    ann["bbox"] = mask_utils.toBbox(ann["segmentation"])

                ann["id"] = id + 1

        self.dataset["annotations"] = result_anns
        self._create_index()

        img_ids_in_result = [ann["image_id"] for ann in result_anns]

        assert set(img_ids_in_result) == (
            set(img_ids_in_result) & set(self.get_img_ids())
        ), "Results do not correspond to current LVIS set."

    def limit_dets_per_image(self, anns, max_dets):
        """
        Bororowed from
        https://github.com/lvis-dataset/lvis-api/blob/master/lvis/results.py#L73
        """
        img_ann = defaultdict(list)
        for ann in anns:
            img_ann[ann["image_id"]].append(ann)

        for img_id, _anns in img_ann.items():
            if len(_anns) <= max_dets:
                continue
            _anns = sorted(_anns, key=lambda ann: ann["score"], reverse=True)
            img_ann[img_id] = _anns[:max_dets]

        return [ann for anns in img_ann.values() for ann in anns]

    def get_top_results(self, img_id, score_thrs):
        """
        Borrowed from
        https://github.com/lvis-dataset/lvis-api/blob/master/lvis/results.py#L86
        """
        ann_ids = self.get_ann_ids(img_ids=[img_id])
        anns = self.load_anns(ann_ids)
        return list(filter(lambda ann: ann["score"] > score_thrs, anns))

    def get_im_joint_attr_cat_to_ann_id(self):
        """
        this func exists for compatibilty with PACO, but is not used in the
        final evaluation. The final evaluation uses object category based
        detections detections since all prediced objects have all attribute
        predictions
        """
        im_cat_to_ann_id = defaultdict(list)
        for ann in self.dataset["annotations"]:
            obj_cat = ann["category_id"]
            if obj_cat not in self.obj_cats:
                continue
            for a, _ in enumerate(ann["attribute_probs"]):
                if (obj_cat, a) in self.pair_to_joint_cat.keys():
                    im_cat_to_ann_id[
                        (ann["image_id"], self.pair_to_joint_cat[obj_cat, a])
                    ].append(ann["id"])
        return im_cat_to_ann_id
