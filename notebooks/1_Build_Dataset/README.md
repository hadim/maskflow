**TODO**: Fllowing is deprecated.

# Build your Training Dataset

## General

`maskflow` does not provide ready to download dataset but instead provides notebooks that can build a various set of dataset for you. Each notebook in this folder will build a different dataset and format it to the COCO format.

The best way to learn is to look directly at the examples:

- [The toy shapes dataset](./Shapes/Shapes.ipynb).
- [The nucleus dataset](./Nucleus/Nucleus.ipynb).

---

## Details

`maskfow` assumes the following:

- A root directory is created for each model/dataset (`ROOT`).
- A data directory called `Data` is created inside the root directory, it contains the following:
    - `train_dataset/`: it contains one PNG image per sample, used for training.
    - `test_dataset/`: it contains one PNG image per sample, used for testing/evaluation.
    - `train_annotations.json`: image annotations of the training dataset following the COCO format.
    - `test_annotations.json`: image annotations of the testing dataset following the COCO format.
- A YAML configuration file describing your model called `config.yaml`.
- A model directory called `Models` which contains one subdirectory by training session.
