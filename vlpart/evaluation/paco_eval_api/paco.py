# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

from lvis import LVIS


class PACO(LVIS):
    def __init__(self, annotation_path):
        """Class for reading PACO annotations.
        Args:
            annotation_path (str): location of annotation file
        """
        super().__init__(annotation_path)

    def get_im_joint_attr_cat_to_ann_id(self):
        """
        Returns a map from im id and joint attr cat to gt annotation ids
        """
        im_cat_to_ann_id = defaultdict(list)
        for ann in self.dataset["annotations"]:
            if len(ann["segmentation"]) == 0:
                continue
            obj_cat = ann["category_id"]
            if obj_cat not in self.obj_cats:
                continue
            for a in ann["attribute_ids"]:
                if (obj_cat, a) in self.pair_to_joint_cat.keys():
                    im_cat_to_ann_id[
                        (ann["image_id"], self.pair_to_joint_cat[obj_cat, a])
                    ].append(ann["id"])

        return im_cat_to_ann_id

    def _create_index(self):
        """
        Overrides https://github.com/lvis-dataset/lvis-api/blob/master/lvis/lvis.py#L38
        to support attributes
        """
        self.logger.info("Creating index.")

        self.img_ann_map = defaultdict(list)

        self.anns = {}
        self.cats = {}
        self.imgs = {}
        self.joint_attr_cats = {}

        # setup metadata to be used by eval and for post processing results
        joint_attr_ids_for_eval = self.dataset["joint_obj_attribute_categories"]
        self.pair_to_joint_cat = {
            (_i["obj"], _i["attr"]): _i["obj-attr"] for _i in joint_attr_ids_for_eval
        }
        self.joint_attr_cat_to_obj_attr = {
            _i["obj-attr"]: (_i["obj"], _i["attr"]) for _i in joint_attr_ids_for_eval
        }
        self.obj_cats = {_i[0] for _i in self.pair_to_joint_cat.keys()}
        self.obj_ids = sorted(
            {x["id"] for x in self.dataset["categories"] if ":" not in x["name"]}
        )
        self.part_ids = sorted(
            {x["id"] for x in self.dataset["categories"] if ":" in x["name"]}
        )
        self.obj_name_to_id = {x["name"]: x["id"] for x in self.dataset["categories"]}
        self.obj_names = {x["id"]: x["name"] for x in self.dataset["categories"]}
        self.attr_names = {x["id"]: x["name"] for x in self.dataset["attributes"]}
        self.joint_cat_to_pair_name = {
            _i["obj-attr"]: (self.obj_names[_i["obj"]], self.attr_names[_i["attr"]])
            for _i in joint_attr_ids_for_eval
        }
        self.part_id_to_obj_id = {
            x["id"]: self.obj_name_to_id[x["name"].split(":")[0]]
            for x in self.dataset["categories"]
            if ":" in x["name"]
        }

        self.attr_cats = {_i[1] for _i in self.pair_to_joint_cat.keys()}

        self.attr_type_to_attr_idxs = self.dataset["attr_type_to_attr_idxs"]
        self.attr_idxs_to_type = {
            _v: k for k, vals in self.attr_type_to_attr_idxs.items() for _v in vals
        }

        # setup categories and annotations
        # self.im_joint_attr_cat_to_ann_id is used to load gt in the eval code
        self.im_joint_attr_cat_to_ann_id = self.get_im_joint_attr_cat_to_ann_id()
        for ann in self.dataset["annotations"]:
            if len(ann["segmentation"]) == 0:
                continue
            self.img_ann_map[ann["image_id"]].append(ann)
            self.anns[ann["id"]] = ann

        for img in self.dataset["images"]:
            self.imgs[img["id"]] = img

        for cat in self.dataset["categories"]:
            self.cats[cat["id"]] = cat

        for cat in self.dataset["joint_obj_attribute_categories"]:
            self.joint_attr_cats[cat["obj-attr"]] = cat

        self.logger.info("Index created.")

    def get_joint_attr_ids(self):
        """Get all joint attr category ids.
        Returns:
            ids (int array): integer array of joint attr category ids
        """
        return list(self.joint_attr_cats.keys())
