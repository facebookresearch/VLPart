# Copyright (c) Meta Platforms, Inc. and affiliates.
import copy
import itertools
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.evaluation.lvis_evaluation import LVISEvaluator

from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from .utils.paco_utils import (
    get_AP_from_precisions,
    get_AP50_from_precisions,
    get_mean_AP,
    heirachrichal_APs,
)


def instances_to_coco_json(instances, img_id):
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    # support for attributes
    has_attrs = instances.has("attribute_probs")
    if has_attrs:
        attribute_probs = instances.attribute_probs

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        if has_attrs:
            result["attribute_probs"] = attribute_probs[k].numpy().tolist()
        results.append(result)
    return results

class PACOEvaluator(LVISEvaluator):
    """
    Evaluate object proposal and instance detection/segmentation outputs using
    LVIS's metrics and evaluation API.
    """

    def __init__(
        self,
        dataset_name,
        tasks=("segm", "bbox"),
        distributed=True,
        output_dir=None,
        eval_attributes=False,
        eval_per=False,
        attr_ap_type="usual",
        *,
        max_dets_per_image=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                "json_file": the path to the LVIS format annotation
            tasks (tuple[str]): tasks that can be evaluated under the given
                configuration. A task is one of "bbox", "segm".
                By default, will infer this automatically from predictions.
            distributed (True): if True, will collect results from all ranks
                for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
            eval_attributes (bool): If true, use PACO eval api; use LVIS eval api otherwise
            attr_ap_type (str): str value in ["usual", "lb", "ub"].
                usual -> usual attr AP, lb -> lower bound attr AP,
                ub -> upper bound attr AP
            max_dets_per_image (None or int): limit on maximum detections per
                image in evaluating AP
                This limit, by default of the LVIS dataset, is 300.
        """
        # if eval_attributes:
        #     from .paco_eval_api import PACO as LVIS
        # else:
        #     from lvis import LVIS
        from .paco_eval_api import PACO as LVIS

        self._logger = logging.getLogger('detectron2.vlpart.evaluation.paco_evaluation')

        if tasks is not None and isinstance(tasks, CfgNode):
            self._logger.warn(
                "COCO Evaluator instantiated using config, this is deprecated behavior."
                " Please pass in explicit arguments instead."
            )
            self._tasks = None  # Infering it from predictions should be better
        else:
            self._tasks = tasks

        self._distributed = distributed
        self._output_dir = output_dir
        self._max_dets_per_image = max_dets_per_image
        self._cpu_device = torch.device("cpu")
        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._lvis_api = LVIS(json_file)
        # Test set json files do not contain annotations (evaluation must be
        # performed using the LVIS evaluation server).
        self._do_evaluation = len(self._lvis_api.get_ann_ids()) > 0
        self.eval_attributes = eval_attributes
        self.attr_ap_type = attr_ap_type
        self.eval_per = eval_per

    def process(self, inputs, outputs):
        """
        Borrows from https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/lvis_evaluation.py#L80
        Change is that we update instances_to_coco_json to support attributes
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"]
                )
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def _eval_predictions(self, predictions):
        """
        Borrows from https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/lvis_evaluation.py#L134
        Change is that update the signature of _evaluate_predictions_on_lvis
        for attriutes
        """
        self._logger.info("Preparing results in the LVIS format ...")
        lvis_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(lvis_results)

        # LVIS evaluator can be used to evaluate results for COCO dataset categories.
        # In this case `_metadata` variable will have a field with
        # COCO-specific category mapping.
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k
                for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in lvis_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]
        else:
            # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
            for result in lvis_results:
                result["category_id"] += 1

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "lvis_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(lvis_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        # change from D2 begins
        for task in sorted(tasks):
            res = _evaluate_predictions_on_lvis(
                self._lvis_api,
                lvis_results,
                task,
                max_dets_per_image=self._max_dets_per_image,
                eval_attributes=self.eval_attributes,
                attr_ap_type=self.attr_ap_type,
                eval_per=self.eval_per,
            )
            self._results[task] = res
        # change from D2 ends


def _evaluate_predictions_on_lvis(
    lvis_gt,
    lvis_results,
    iou_type,
    max_dets_per_image=None,
    eval_attributes=False,
    attr_ap_type="usual",
    eval_per=False,
):
    """
    Runs either regular LVIS eval or PACO eval depending on whether
    eval_attributes is set to true. Post process and reports the results
    Args:
        iou_type (str): segm of bbox
        max_dets_per_image (None or int): limit on maximum detections per image
            in evaluating AP
            This limit, by default of the LVIS dataset, is 300.
        eval_attributes: if True, use PACO eval, else use LVIS
        attr_ap_type: str in ["usual", "lb", "ub"] to select whether to report
            usual AP or lower bound or upper bound, for attrs
    """
    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
    }[iou_type]

    # logger = logging.getLogger(__name__)
    logger = logging.getLogger('detectron2.vlpart.evaluation.paco_evaluation')

    if len(lvis_results) == 0:  # TODO: check if needed
        logger.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    if iou_type == "segm":
        lvis_results = copy.deepcopy(lvis_results)
        # When evaluating mask AP, if the results contain bbox, LVIS API will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in lvis_results:
            c.pop("bbox", None)

    if max_dets_per_image is None:
        max_dets_per_image = 300  # Default for LVIS dataset

    logger.info(f"Evaluating with max detections per image = {max_dets_per_image}")
    if eval_attributes:
        from .paco_eval_api import PACOEval as LVISEval, PACOResults as LVISResults

        lvis_results = LVISResults(lvis_gt, lvis_results, max_dets=max_dets_per_image)
        lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type, attr_ap_type=attr_ap_type)
    else:
        from lvis import LVISEval, LVISResults

        lvis_results = LVISResults(lvis_gt, lvis_results, max_dets=max_dets_per_image)
        lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)

    lvis_eval.run()
    lvis_eval.print_results()

    # post process and save results

    precisions = lvis_eval.eval["precision"]
    # all 531 classes
    all_obj_names = lvis_eval.lvis_gt.obj_names
    # excludes cat ids corressponding to object-parts
    obj_cat_ids = []
    # 200 semantic part classes to object-part cats
    part_id_to_obj_part_ids = defaultdict(list)
    for x in lvis_eval.lvis_gt.dataset["categories"]:
        if ":" in x["name"]:
            part_id_to_obj_part_ids[x["name"].split(":")[-1]].append(x["id"])
        else:
            obj_cat_ids.append(x["id"])
    # map cat ids to indices in eval results
    sorted_cats = sorted(lvis_eval.lvis_gt.dataset["categories"], key=lambda x: x["id"])
    obj_cats_to_cont_id_eval = {cat["id"]: _i for _i, cat in enumerate(sorted_cats)}

    # report AP for 75 object classes
    results_processed = {}
    obj_results = []
    obj_results_per_class = {}
    obj_results_ap50_per_class = {}
    for obj_cat in obj_cat_ids:
        idx = obj_cats_to_cont_id_eval[obj_cat]
        ap = get_AP_from_precisions(precisions, idx)
        obj_results.append(float(ap * 100))
        obj_results_per_class[all_obj_names[obj_cat]] = ap * 100
        ap50 = get_AP50_from_precisions(precisions, idx)
        obj_results_ap50_per_class[all_obj_names[obj_cat]] = ap50 * 100

    results_processed["obj-AP"] = np.around(
        get_mean_AP(list(obj_results_per_class.values())), 3)
    results_processed["obj-AP50"] = np.around(
        get_mean_AP(list(obj_results_ap50_per_class.values())), 3)
    if eval_per:
        results_processed["per-obj-AP"] = obj_results_per_class

    # report AP for 200 part classes
    part_results_per_class = {}
    part_results_ap50_per_class = {}
    for part, obj_part_ids in part_id_to_obj_part_ids.items():
        results_for_part = []
        results_ap50_for_part = []
        for _id in obj_part_ids:
            idx = obj_cats_to_cont_id_eval[_id]
            ap = get_AP_from_precisions(precisions, idx)
            results_for_part.append(float(ap * 100))

            ap50 = get_AP50_from_precisions(precisions, idx)
            results_ap50_for_part.append(float(ap50 * 100))

        part_results_per_class[part] = get_mean_AP(results_for_part)
        part_results_ap50_per_class[part] = get_mean_AP(results_ap50_for_part)

    overall_part_res = np.array(list(part_results_per_class.values()))
    results_processed["obj-part-AP-heirarchical"] = np.around(
        np.mean(overall_part_res[overall_part_res > -1]), 3)
    overall_ap50_part_res = np.array(list(part_results_ap50_per_class.values()))
    results_processed["obj-part-AP50-heirarchical"] = np.around(
        np.mean(overall_ap50_part_res[overall_ap50_part_res > -1]), 3)
    if eval_per:
        results_processed["per-part-AP"] = part_results_per_class

    if eval_attributes:
        # report AP for attrs
        results_heirarchical = heirachrichal_APs(lvis_eval)
        results_processed["attributes-heirarchical"] = results_heirarchical

    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}
    results["post-processed"] = results_processed

    logger.info(
        "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
    )
    return results