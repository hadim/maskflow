import datetime
from pathlib import Path
import copy
import random
import json
import numpy as np

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader

from .cococreator import create_image_info
from .cococreator import create_annotation_info


class DatasetCatalog:
    DATASETS = {
        "train_dataset": {
            "root": "train_dataset",
            "ann_file": "train_annotations.json",
        },
        "test_dataset": {
            "root": "test_dataset",
            "ann_file": "test_annotations.json",
        },
    }

    @staticmethod
    def get(name):
        data_dir = cfg["DATA_DIR"]
        if data_dir is None:
            raise Exception("You need to set `config['DATA_DIR']")
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(root=Path(data_dir) / attrs["root"],
                    ann_file=Path(data_dir) / attrs["ann_file"])
        return dict(factory="COCODataset", args=args)


def get_data_loader(config, data_dir, is_train=True):
    config['DATA_DIR'] = data_dir
    data_loader = make_data_loader(config, is_train=is_train)
    # FIXME: maskrcnn-benchmark returns a `DataLoader` or a list of
    # `DataLoader` depending on the `is_train` value. We should only return
    # a single DataLoader.
    if is_train:
        return data_loader
    else:
        return data_loader[0]


def get_base_annotations(class_names, supercategory=""):

    categories = get_categories(class_names, supercategory=supercategory)

    base_annotations = {
        "info": {"description": "Toy Shapes Dataset",
                 "url": "https://github.com/hadim/maskflow",
                 "version": "0.1.0",
                 "year": 2018,
                 "contributor": "hadim",
                 "date_created": datetime.datetime.utcnow().isoformat(' ')
                },
        "licenses": {"id": 1,
                     "name": "Attribution-NonCommercial-ShareAlike License",
                     "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                    },
        "categories": categories,
        "images": [],
        "annotations": []
    }
    return copy.deepcopy(base_annotations)


def get_categories(class_names, supercategory=""):
    return [dict(id=i + 1, name=name, supercategory=supercategory) for i, name in enumerate(class_names)]


def get_annotations(image_id, basename, image, mask, class_ids):
    
    assert image.shape[:2] == mask.shape[1:], "Mask needs to have the same size as the image."
    
    image_info = create_image_info(image_id, basename, image.shape[:2])

    image_annotations = []
    for binary_mask, class_id in zip(mask, class_ids):
        category_info = {'id': int(class_id), 'is_crowd': False}

        annotation_info = create_annotation_info(
            random.getrandbits(24), image_id, category_info, binary_mask,
            image.shape[:2], tolerance=0)
        if annotation_info:
            image_annotations.append(annotation_info)

    return image_info, image_annotations


def save_annotations(annotations, annotation_path):
    with open(annotation_path, 'w') as f:
        json.dump(annotations, f)

        

def crop_image(image, masks, class_ids, final_size):
    """Crop image and mask to final_size if needed. It add zeros values if imge is
    smaller and it crop it if the image is bigger than final_size. Remove empty masks
    after crop if needed. Returns None if no object is left after crop.
    """
    
    assert image.shape[:2] == masks.shape[-2:], "Image and masks need to have the same size."
    assert class_ids.shape[0] == masks.shape[0], "Class ids and masks need to have the same number of objects."
    
    w, h = image.shape[:2]

    # We do nothing
    if w == final_size and h == final_size:
        return image, masks, class_ids

    crop_w = final_size - w
    crop_h = final_size - h
    
    new_w = w + crop_w if crop_w > 0 else final_size
    new_h = h + crop_h if crop_h > 0 else final_size
    
    new_image = np.zeros((new_w, new_h, image.shape[-1]))
    new_image[:w, :h] = image[:new_w, :new_h]
    new_image = new_image.astype(image.dtype)

    new_masks = np.zeros((masks.shape[0], new_w, new_h))
    new_masks[:, :w, :h] = masks[:, :new_w, :new_h]
    new_masks = new_masks.astype(masks.dtype)
    
    # Check mask that still contain an object.
    to_keep = np.where(new_masks.sum(axis=-1).sum(axis=-1) > 0)[0]
    new_masks = new_masks[to_keep]
    class_ids = class_ids[to_keep]
    
    if new_masks.shape[0] == 0:
        return None, None, None
    else:
        return new_image, new_masks, class_ids
