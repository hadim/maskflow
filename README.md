# `maskflow`: Object Detection and Segmentation

A Python package for [Mask RCNN semantic segmentation](https://arxiv.org/abs/1703.06870). It has been freely and strongly inspired from this [Matterport Tensorflow implementation of MaskRCNN](https://github.com/matterport/Mask_RCNN).

It comes with a [Fiji plugin](https://github.com/hadim/maskflow-fiji) that can run prediction using a model trained with this package.

## Usage

- Setup the Anaconda environment:

```
conda env create -f environment.yml
```

- [Navigate in the notebooks](./notebooks/1_Build_Dataset/README.md) in order to train and pack your Tensorflow model.

## License

Under BSD license. See [LICENSE](LICENSE).

## Authors

- Hadrien Mary <hadrien.mary@gmail.com>
