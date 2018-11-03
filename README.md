# `maskflow`: Object Detection and Segmentation for Cell Biology

A Python package for [Mask RCNN semantic segmentation](https://arxiv.org/abs/1703.06870) in microsopy images. It uses [the following implementation](https://github.com/facebookresearch/maskrcnn-benchmark).

It comes with a [Fiji plugin](https://github.com/hadim/maskflow-fiji) that can run prediction using a model trained with this package.

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
