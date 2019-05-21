# Build your Training Dataset

## Introduction

`maskflow` provides notebooks that can build a various set of dataset for you.

We use [Tensorflow TFRecords](https://www.tensorflow.org/tutorials/load_data/tf_records) files for input dataset. The various notebooks provided here are in charge of converting a dataset to TFRecords files. So, as long a dataset follows the TFRecords structure (see Details), it should be straightforward to train it with `maskflow`.

The best way to learn is to look directly at the examples:

- [The toy shapes dataset](./Shapes/Shapes.ipynb): tiny auto-generated dataset used during development or for teaching purpose.
- [The nucleus dataset](./Nucleus/Nucleus.ipynb): from the Kaggle 2018 Data Science Bowl.

## Details

To train a dataset you need:

- A YAML configuration file describing your model `config.yaml`.
- TFRecord files previously generated: `train.tfrecords` and `test.tfrecords`.

Here is how a TFRecord file is structured:

```python
feature_description = {}

# Image features
feature_description['image/height'] = tf.io.FixedLenFeature([], tf.int64)
feature_description['image/width'] = tf.io.FixedLenFeature([], tf.int64)
feature_description['image/channel'] = tf.io.FixedLenFeature([], tf.int64)
feature_description['image/filename'] = tf.io.FixedLenFeature([], tf.string)
feature_description['image/source_id'] = tf.io.FixedLenFeature([], tf.string)
feature_description['image/encoded'] = tf.io.FixedLenFeature([], tf.string)
feature_description['image/key/sha256'] = tf.io.FixedLenFeature([], tf.string)
feature_description['image/format'] = tf.io.FixedLenFeature([], tf.string)

# Object features
feature_description['image/object/bbox/x'] = tf.io.VarLenFeature(tf.float32)
feature_description['image/object/bbox/y'] = tf.io.VarLenFeature(tf.float32)
feature_description['image/object/bbox/width'] = tf.io.VarLenFeature(tf.float32)
feature_description['image/object/bbox/height'] = tf.io.VarLenFeature(tf.float32)
feature_description['image/object/class/text'] = tf.io.VarLenFeature(tf.string)
feature_description['image/object/class/label'] = tf.io.VarLenFeature(tf.int64)
feature_description['image/object/masks'] = tf.io.VarLenFeature(tf.string)
feature_description['image/object/mask_format'] = tf.io.FixedLenFeature([], tf.string)
```