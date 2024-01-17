from detectron2.data.datasets import register_coco_instances
import os

DATA_SET_NAME = "dolphin"
DATA_SET_ROOT = os.path.join(os.path.expanduser("~"), "masters-thesis", "datasets", "ndd20", "coco")
DATA_SET_IMAGES_DIR_PATH = os.path.join(DATA_SET_ROOT, "images")

# # TRAIN SET
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "train_1500.coco.json")

register_coco_instances(
    name=TRAIN_DATA_SET_NAME,
    metadata={},
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH,
    image_root=DATA_SET_IMAGES_DIR_PATH
)

# # VAL SET
VAL_DATA_SET_NAME = f"{DATA_SET_NAME}-val"
VAL_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "val_300.coco.json")

register_coco_instances(
    name=VAL_DATA_SET_NAME,
    metadata={},
    json_file=VAL_DATA_SET_ANN_FILE_PATH,
    image_root=DATA_SET_IMAGES_DIR_PATH
)