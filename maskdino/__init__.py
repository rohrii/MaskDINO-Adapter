# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------------
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskdino_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.detr_dataset_mapper import DetrDatasetMapper

from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
from .data.dataset_mappers.zerowaste_dataset_mapper import ZerowasteDatasetMapper
from .data.dataset_mappers.dolphin_dataset_mapper import DolphinDatasetMapper
from .data.dataset_mappers.xray_waste_dataset_mapper import XrayWasteDatasetMapper

# models
from .maskdino import MaskDINO
# from .data.datasets_detr import coco
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
# util
from .utils import box_ops, misc, utils
