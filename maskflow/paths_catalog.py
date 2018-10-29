from pathlib import Path

from maskrcnn_benchmark.config import cfg


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
