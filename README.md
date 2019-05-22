# Maskflow: Object Detection and Segmentation for Cell Biology
[![Build Status](https://travis-ci.com/hadim/maskflow.svg?branch=master)](https://travis-ci.com/hadim/maskflow)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/hadim/maskflow/master?urlpath=lab/tree/notebooks)

**WARNING: this project is in development. Come back later for an alpha version you can play with.**

Maskflow is a Python framework for **instance object detetion and segmentation** using neural networks. It is focused on processing biological images but aims to be general enough to process any kind of dataset.

The framework aims to be:

- **easy-to-use**: Any person with a minimal Python experience should be able to train and deploy a model.
- **flexible**: While we implement a default model that works well we want to allow the creation of new models easily by just swapping sub-models (*replace the ResNet backbone by a DenseNet backbone for example*).
- **universal**: You should be able to predict labels, bounding boxes, masks. Moreover why do we have to restrict to those when you can predict any kind of polygon describing your objects (*polygon detections is not yet implemented*)?
- **easy-to-deploy**: Deploy your model using Docker and run inference by sending REST calls to it.
- **test-driven**: Maskflow development is test-driven in order to provide a robust and stable API.

As Maskflow is originally designed for biological images, it comes with a [Fiji plugin](https://github.com/hadim/maskflow-fiji) that can run a model previously trained (*the plugin will be available very soon*).

We use **Python** >= 3.6 an **TensorFlow** >= 2.0.

## Trained Models

[Check the notebooks](./notebooks/1_Build_Dataset/README.md) to train your own dataset.

In addition to this, Maskflow comes with multiples ready-to-build-and-train datasets. The [notebooks](./notebooks) provide push-and-start code to build and train a model.

- [Shapes](./notebooks/1_Build_Dataset/Shapes/Shapes.ipynb): A "toy" dataset that can be trained very quickly for debugging and educational purpose.
- [Nucleus](./notebooks/1_Build_Dataset/Nucleus/Nucleus.ipynb): A dataset composed of cell nucleus with various image acquisition modalities.
- [Synthetic_Cells](./notebooks/1_Build_Dataset/Synthetic_Cells/Synthetic_Cells.ipynb): Virtual cell image generated *in silico*.
- [Microtubule](./notebooks/1_Build_Dataset/Microtubule/Microtubule.ipynb): Dataset consists of various simulated *in vitro* microtubule images.

## Usage

We suggest you to use Anaconda as a Python distribution.

```bash
# Create a Python environement.
conda create -n my_env python

# Activate it
conda activate my_env

# Install maskflow dependencies
conda env create -f environment.yml
```

Tensorflow needs to be installed with Pip at the moment. Check the [`environment.yml`](./environment.yml) file for the pip commands to install Tensorflow.

Then you can [navigate to the notebooks](./notebooks/1_Build_Dataset/README.md) in order to train and pack your model. Maskflow is also a perfectly valid Python package that you can install via pip:

```bash
pip install maskflow
# or
pip install https://github.com/hadim/maskflow/archive/master.zip
```

## License

Under BSD license. See [LICENSE](LICENSE).

## Authors

- Hadrien Mary <hadrien.mary@gmail.com>
