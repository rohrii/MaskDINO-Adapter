from detectron2.data.datasets import register_coco_instances
import os

DATA_SET_NAME = "zerowaste"
DATA_SET_ROOT = os.path.join(os.path.expanduser("~"), "masters-thesis", "datasets", "zerowaste-f")

# # TRAIN SET
TRAIN_IMAGES_DIR_PATH = os.path.join(DATA_SET_ROOT, "train", "data")
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
# TRAIN_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "train_20.coco.json")
TRAIN_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "train_clean.json")

register_coco_instances(
    name=TRAIN_DATA_SET_NAME,
    metadata={},
    json_file=TRAIN_ANN_FILE_PATH,
    image_root=TRAIN_IMAGES_DIR_PATH
)

# # VAL SET
VAL_IMAGES_DIR_PATH = os.path.join(DATA_SET_ROOT, "val", "data")
VAL_DATA_SET_NAME = f"{DATA_SET_NAME}-val"
# VAL_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "val_20.coco.json")
VAL_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "val_clean.json")

register_coco_instances(
    name=VAL_DATA_SET_NAME,
    metadata={},
    json_file=VAL_ANN_FILE_PATH,
    image_root=VAL_IMAGES_DIR_PATH
)