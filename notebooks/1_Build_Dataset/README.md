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
- TFRecord files previously generated: `train.tfrecord` and `test.tfrecord`.

Here is how a TFRecord file is structured:

```python
feature_dict = {}

# Image features
feature_dict['image/height'] = maskflow.dataset.int64_feature(image_height)
feature_dict['image/width'] = maskflow.dataset.int64_feature(image_width)
feature_dict['image/filename'] = maskflow.dataset.bytes_feature(filename.encode('utf8'))
feature_dict['image/source_id'] = maskflow.dataset.bytes_feature(str(image_id).encode('utf8'))
feature_dict['image/key/sha256'] = maskflow.dataset.bytes_feature(key.encode('utf8'))
feature_dict['image/encoded'] = maskflow.dataset.bytes_feature(encoded_image)
feature_dict['image/format'] = maskflow.dataset.bytes_feature(image_format.encode('utf8'))

# Object features
feature_dict['image/object/bbox/xmin'] = maskflow.dataset.float_list_feature(xmin)
feature_dict['image/object/bbox/xmax'] = maskflow.dataset.float_list_feature(xmax)
feature_dict['image/object/bbox/ymin'] = maskflow.dataset.float_list_feature(ymin)
feature_dict['image/object/bbox/ymax'] = maskflow.dataset.float_list_feature(ymax)
feature_dict['image/object/class/text'] = maskflow.dataset.float_list_feature(category_names)
feature_dict['image/object/class/label'] = maskflow.dataset.int64_list_feature(category_ids)
feature_dict['image/object/mask'] = maskflow.dataset.bytes_list_feature(encoded_mask_png)

# Features currently not used
#feature_dict['image/object/is_crowd'] = maskflow.dataset.int64_list_feature(is_crowd)
#feature_dict['image/object/area'] = maskflow.dataset.float_list_feature(area)
#feature_dict['image/caption'] = maskflow.dataset.bytes_list_feature(captions)

example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
```