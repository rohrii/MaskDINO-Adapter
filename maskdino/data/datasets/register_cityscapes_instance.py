from detectron2.data.datasets.cityscapes import load_cityscapes_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from cityscapesscripts.helpers.labels import labels

import os


instance_labels = [label for label in labels if label.hasInstances and not label.ignoreInEval]
thing_classes = [label.name for label in instance_labels]
thing_dataset_id_to_contiguous_id = {label.id: i for i, label in enumerate(instance_labels)}

ROOT_PATH = os.path.join(os.path.expanduser("~"), "datasets/cityscapes/data")


# TRAINING

TRAIN_NAME = "cityscapes_train"
TRAIN_IMAGE_DIR = os.path.join(ROOT_PATH, "leftImg8bit/train")
TRAIN_ANNOTATION_DIR = os.path.join(ROOT_PATH, "gtFine/train")

train_fn = lambda: load_cityscapes_instances(TRAIN_IMAGE_DIR, TRAIN_ANNOTATION_DIR)

DatasetCatalog.register(TRAIN_NAME, train_fn)

MetadataCatalog.get(TRAIN_NAME).set(
    thing_classes=thing_classes,
    thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
    gt_dir=TRAIN_ANNOTATION_DIR,
    evaluator_type="cityscapes_instance",
)


# VALIDATION

VAL_NAME = "cityscapes_val"
VAL_IMAGE_DIR = os.path.join(ROOT_PATH, "leftImg8bit/val")
VAL_ANNOTATION_DIR = os.path.join(ROOT_PATH, "gtFine/val")

val_fn = lambda: load_cityscapes_instances(VAL_IMAGE_DIR, VAL_ANNOTATION_DIR)

DatasetCatalog.register(VAL_NAME, val_fn)

MetadataCatalog.get(VAL_NAME).set(
    thing_classes=thing_classes,
    thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
    gt_dir=VAL_ANNOTATION_DIR,
    evaluator_type="cityscapes_instance",
)
