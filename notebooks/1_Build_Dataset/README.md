# Training dataset Generation

To train your dataset on `maskflow` you have **two choices**. Note that for inference, more classic Numpy arrays are used as input.

## TFRecord

This is preferred way since, `maskflow` comes with convenient functions to convert your dataset to [`TFRecord` files](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset). 

For an example, you can check the [Shapes dataset](./Shapes/Shapes.ipynb).

Here is the the features map used to format TFRecord elements:

```python
{
# An integer that identify the image.
"image/id": tf.FixedLenFeature([], tf.int64),
# The name of the image.
"image/basename": tf.FixedLenFeature([], tf.string),
# The width of the image (W).
"image/width": tf.FixedLenFeature([], tf.int64),
# The height of the image (H).
"image/height": tf.FixedLenFeature([], tf.int64),
# The number of channels of the image.
"image/channel": tf.FixedLenFeature([], tf.int64),
# The number of objects in the image (N).
"image/n_objects": tf.FixedLenFeature([], tf.int64),
# The image bytes as PNG.
"image/image_bytes": tf.FixedLenFeature([], tf.string),
# A sparse format of the mask array representing the indices of positive pixels.
# A 1D array that is reshaped to [-1, 3], and then used by tf.sparse_tensor_to_dense(),
# to reconsitute the mask array with the following shape [N, W, H].
"image/masks_indices": tf.VarLenFeature(tf.int64),
# The class indices of the objects: [N, ]. The order of the objects need to be the same
# as for "image/masks_indices" once reconstituted by tf.sparse_tensor_to_dense().
"image/class_ids": tf.VarLenFeature(tf.int64)
}
```

## tf.data.Dataset

You can also provide your own `tf.data.Dataset` object. Here is the nestted signature you need to respect:

```python

def build_my_dataset():
    # Do things...
    return images, {"masks": masks, "class_ids": class_ids, "image_id": features['image/id']}
```

where:

- `images`: The image array, `[W, H, C]`.
- `masks`: The mask array, `[N, W, H]`, where `N` is the number of objects in the image.
- `class_ids`: The different class id of the objects, `[N,]`.
- `image_id`: the id of the image, `[]`.