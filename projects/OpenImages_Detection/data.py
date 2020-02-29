# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from detectron2/data/datasets/coco.py by Claylau.

import os
import logging
import csv
from PIL import Image

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)


def load_openimages_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    img_ids = []
    is_groupof = []
    img_sizes = []
    boxes = []
    with open(json_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        reader = iter(reader)
        next(reader) # skip the first line.
        for instance in reader:
            # filter IsDepiction
            if instance[11] == '1':
                continue
            img_id = instance[0]
            im = Image.open(image_root + '/' + img_id + '.jpg')
            w, h = im.size
            img_ids.append(img_id)
            img_sizes.append((h, w))
            x_min, x_max = float(instance[4])*(w-1), float(instance[5])*(w-1)
            y_min, y_max = float(instance[6])*(w-1), float(instance[7])*(h-1)
            boxes.append([x_min, y_min, x_max, y_max])
            # instance[10] IsGroupOf, unused.
            del im

    dataset_dicts = []
    unique_img_ids = [0]
    for i in range(len(boxes)):
        img_id = img_ids[i]
        if img_id != unique_img_ids[-1]:
            unique_img_ids.append(img_id)
            record = {}
            record["file_name"] = os.path.join(image_root, img_id+".jpg")
            record["height"] = img_sizes[i][0]
            record["width"] = img_sizes[i][1]
            record["image_id"] = img_id
            record["annotations"] = []
            obj = {}
            # obj["iscrowd"] = is_groupof[i]
            obj["bbox"] = boxes[i]
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            obj["category_id"] = 1
            record["annotations"].append(obj)
            dataset_dicts.append(record)
        else:
            unique_img_ids.append(img_id)
            obj = {}
            # obj["iscrowd"] = is_groupof[i]
            obj["bbox"] = boxes[i]
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            obj["category_id"] = 1
            dataset_dicts[-1]["annotations"].append(obj)
    
    logger.info("Loaded {} images in COCO format from {}".format(len(dataset_dicts), json_file))

    return dataset_dicts


def register_openimages_instances(name, json_file, image_root):
    """
    Args:
        name (str): the name that identifies a dataset, e.g. "openimages_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_openimages_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco",
        thing_classes=["__not-pen__", "pen"]
    )


def register_all_openimages(root="datasets"):
    _PREDEFINED_SPLITS_OPENIMAGES = {}
    _PREDEFINED_SPLITS_OPENIMAGES["openimages"] = {
        "pen_train": ("images/train", "train-annotations-bbox.csv"),
        "pen_val": ("images/val", "val-annotations-bbox.csv"),
        "pen_test": ("images/test", "test-annotations-bbox.csv"),

    }
    for _, splits_per_dataset in _PREDEFINED_SPLITS_OPENIMAGES.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_openimages_instances(
                key,
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )
