import datetime
from pathlib import Path
import copy
import random
import json

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
    data_loader = data_loader[0] if isinstance(data_loader, list) else data_loader
    return data_loader
  

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
    return [dict(id=i+1, name=name, supercategory=supercategory) for i, name in enumerate(class_names)]


def get_annotations(image_id, basename, image, mask, class_ids):
    image_info = create_image_info(image_id, basename, image.shape)
    
    image_annotations = []
    for binary_mask, class_id in zip(mask, class_ids):
        category_info = {'id': int(class_id), 'is_crowd': False}
        
        annotation_info = create_annotation_info(
            random.getrandbits(24), image_id, category_info, binary_mask,
            image.shape[:-1], tolerance=0)
        if annotation_info:
            image_annotations.append(annotation_info)
            
    return image_info, image_annotations
    
    
def save_annotations(annotations, annotation_path):
    with open(annotation_path, 'w') as f:
        json.dump(annotations, f)