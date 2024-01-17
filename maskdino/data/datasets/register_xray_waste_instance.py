from detectron2.data.datasets import register_coco_instances
import os

DATA_SET_NAME = "xray-waste"
DATA_SET_ROOT = os.path.join(os.path.expanduser("~"), "datasets", "WIXray-master")


# # TRAIN SET
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "annotations", "instances_train2017.json")
TRAIN_DATA_SET_IMAGE_ROOT = os.path.join(DATA_SET_ROOT, "images", "train2017")
register_coco_instances(
    name=TRAIN_DATA_SET_NAME,
    metadata={},
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH,
    image_root=TRAIN_DATA_SET_IMAGE_ROOT
)

# # VAL SET
VAL_DATA_SET_NAME = f"{DATA_SET_NAME}-val"
VAL_DATA_SET_ANN_FILE_PATH = os.path.join(DATA_SET_ROOT, "annotations", "instances_val2017.json")
VAL_DATA_SET_IMAGE_ROOT = os.path.join(DATA_SET_ROOT, "images", "val2017")

register_coco_instances(
    name=VAL_DATA_SET_NAME,
    metadata={},
    json_file=VAL_DATA_SET_ANN_FILE_PATH,
    image_root=VAL_DATA_SET_IMAGE_ROOT
)