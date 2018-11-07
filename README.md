# `maskflow`: Object Detection and Segmentation for Cell Biology
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/hadim/maskflow/master?urlpath=lab/tree/notebooks)
[![Build Status](https://travis-ci.com/hadim/maskflow.svg?branch=master)](https://travis-ci.com/hadim/maskflow)

A Python package for [Mask RCNN semantic segmentation](https://arxiv.org/abs/1703.06870) in microsopy images. It uses [the following implementation](https://github.com/facebookresearch/maskrcnn-benchmark).

It comes with a [Fiji plugin](https://github.com/hadim/maskflow-fiji) that can run prediction using a model trained with this package (not ready yet).

## Trained Models

`maskflow` allows you to easily trained [your own dataset for object detection and segmentation](./notebooks/1_Build_Dataset/README.md).

In addition to this, it comes with for the moment 4 ready to build and train dataset. The [notebooks](./notebooks) provide push-and-start code to build and train a model.

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

## Future

In the future, we would like to provide a Fiji plugin that will download a pretrained model and run inference on it. [A proof of concept is available](https://github.com/hadim/maskflow-fiji) but is not compatible with `maskflow` for the moment.

## License

Under BSD license. See [LICENSE](LICENSE).

## Authors

- Hadrien Mary <hadrien.mary@gmail.com>
