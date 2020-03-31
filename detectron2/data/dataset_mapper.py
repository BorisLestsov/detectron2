# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from . import detection_utils as utils
from . import transforms as T

from fvcore.transforms.transform import (
    HFlipTransform,
    NoOpTransform,
    TransformList,
)

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True, use_cons=False, use_hard_aug=False, use_anno_modify=False):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.add_trans = []
        self.add_trans.append(T.CTAug())
        self.add_trans.append(T.RandomFlip(0.5))
        # self.add_trans.append(T.RandomContrast(0.5, 2))
        # self.add_trans.append(T.RandomBrightness(0.5, 2))
        # self.add_trans.append(T.RandomSaturation(0.5, 2))
        # self.add_trans.append(T.RandomLighting(1))
        self.cons     = use_cons

        tfm_gens = utils.build_transform_gen(cfg, is_train, hard_aug=use_hard_aug, anno_modify=use_anno_modify)
        self.static_tfm_gens, self.dynamic_tfm_gens = tfm_gens[:1], tfm_gens[1:]

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict_orig):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        image_orig = utils.read_image(dataset_dict_orig["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict_orig, image_orig)
        image_orig, transforms_stat = T.apply_transform_gens(self.static_tfm_gens, image_orig)

        # USER: Write your own image loading if it's not from a file

        image = image_orig.copy()

        if "annotations" not in dataset_dict_orig:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict_orig["annotations"]),
                )
                image = crop_tfm.apply_image(image)

            image, transforms = T.apply_transform_gens(self.dynamic_tfm_gens, image)
            transforms = transforms_stat + transforms

            if self.crop_gen:
                transforms = crop_tfm + transforms

            image_shape = image.shape[:2]  # h, w

        image_orig = image.copy()

        dup = 2 if self.cons else 1
        ret = []
        for dup_i in range(dup):
            dataset_dict = copy.deepcopy(dataset_dict_orig)  # it will be modified by code below
            image = image_orig.copy()

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
            # Can use uint8 if it turns out to be slow some day

            # USER: Remove if you don't use pre-computed proposals.
            if self.load_proposals:
                utils.transform_proposals(
                    dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
                )

            if not self.is_train:
                dataset_dict.pop("annotations", None)
                dataset_dict.pop("sem_seg_file_name", None)
                return dataset_dict

            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    if not self.mask_on:
                        anno.pop("segmentation", None)
                    if not self.keypoint_on:
                        anno.pop("keypoints", None)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    utils.transform_instance_annotations(
                        obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                    )
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = utils.annotations_to_instances(
                    annos, image_shape, mask_format=self.mask_format
                )
                # Create a tight bounding box from masks, useful when image is cropped
                if self.crop_gen and instances.has("gt_masks"):
                    instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
                dataset_dict["instances"] = utils.filter_empty_instances(instances)

            # USER: Remove if you don't do semantic/panoptic segmentation.
            if "sem_seg_file_name" in dataset_dict:
                with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                    sem_seg_gt = Image.open(f)
                    sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
                sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
                sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
                dataset_dict["sem_seg"] = sem_seg_gt


            # CONSISTENCY_SAMPLE
            image_orig, transforms_add = T.apply_transform_gens(self.add_trans, image_orig)
            transforms += transforms_add

            if dup_i > 0:
                rev_trans = []
                for i, tr in enumerate(transforms_add.transforms[::-1]):
                    if isinstance(tr, HFlipTransform):
                        rev_trans.append(HFlipTransform(tr.width))
                rev_trans = TransformList(rev_trans)
                dataset_dict["rev_tr"] = rev_trans

            ret.append(dataset_dict)

        return ret
