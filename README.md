# `maskflow`: Object Detection and Segmentation for Cell Biology
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/hadim/maskflow/master?urlpath=lab/tree/notebooks)
[![Build Status](https://travis-ci.com/hadim/maskflow.svg?branch=master)](https://travis-ci.com/hadim/maskflow)

A Python package for semantic segmentation in biological microsopy images. The model used is [Mask R-CNN](https://arxiv.org/abs/1703.06870) and is implemented in TensorFlow 2.0.

It comes with a [Fiji plugin](https://github.com/hadim/maskflow-fiji) that can run a model trained with this package.

**IMPORTANT: This Python library is a proof of concept and in a very early stage. I hope to be able to continue the development at some point.**

## Trained Models

`maskflow` allows you to easily trained [your own dataset for object detection and segmentation](./notebooks/1_Build_Dataset/README.md).

In addition to this, it comes with 4 ready-to-build-and-train datasets. The [notebooks](./notebooks) provide push-and-start code to build and train a model.

- [Shapes](./notebooks/1_Build_Dataset/Shapes/Shapes.ipynb): A "toy" dataset that can be trained very quickly for debugging and educational purpose.
- [Nucleus](./notebooks/1_Build_Dataset/Nucleus/Nucleus.ipynb): A dataset composed of cell nucleus with various image acquisition modalities.
- [Synthetic_Cells](./notebooks/1_Build_Dataset/Synthetic_Cells/Synthetic_Cells.ipynb): Virtual cell image generated *in silico*.
- [Microtubule](./notebooks/1_Build_Dataset/Microtubule/Microtubule.ipynb): Dataset consists of various simulated *in vitro* microtubule images.

## Usage

- Setup the Anaconda environment:

```
conda env create -f environment.yml
```

- [Navigate in the notebooks](./notebooks/1_Build_Dataset/README.md) in order to train and pack your model.

## License

Under BSD license. See [LICENSE](LICENSE).

## Authors

- Hadrien Mary <hadrien.mary@gmail.com>
