from io import BytesIO
import tensorflow as tf
import numpy as np
from PIL import Image


def _array_to_png(arr):
    with BytesIO() as image_bytes:
        im = Image.fromarray(arr)
        im.save(image_bytes, format="png")
        image_bytes = image_bytes.getvalue()
    return image_bytes


def _mask_to_indices(mask):
    return np.argwhere(mask == 1)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_tf_example(i, basename, image, mask, class_ids):
    
    n_channel = image.shape[2] if len(image.shape) == 3 else 1
    
    features = {"image/id": _int64_feature(i),
                "image/basename": _bytes_feature(basename.encode("utf-8")),
                "image/width": _int64_feature(image.shape[0]),
                "image/height": _int64_feature(image.shape[1]),
                "image/n_objects": _int64_feature(mask.shape[0]),
                "image/image_bytes": _bytes_feature(_array_to_png(image)),
                "image/masks_indices": _int64_list_feature(_mask_to_indices(mask).flatten()),
                "image/class_ids": _int64_list_feature(class_ids)}
    
    return tf.train.Example(features=tf.train.Features(feature=features))


def decode_tfrecord(serialized_example):
    """Parses features and labels from the given `serialized_example`."""
    features_map = {"image/id": tf.FixedLenFeature([], tf.int64),
                    "image/basename": tf.FixedLenFeature([], tf.string),
                    "image/width": tf.FixedLenFeature([], tf.int64),
                    "image/height": tf.FixedLenFeature([], tf.int64),
                    "image/n_objects": tf.FixedLenFeature([], tf.int64),
                    "image/image_bytes": tf.FixedLenFeature([], tf.string),
                    "image/masks_indices": tf.VarLenFeature(tf.int64),
                    "image/class_ids": tf.VarLenFeature(tf.int64)}

    features = tf.parse_single_example(serialized_example, features_map)

    # Decode the image (we assume PNG)
    image = tf.image.decode_png(features["image/image_bytes"])
    
    if image.ndim == 2:
        image = tf.expand_dims(image, axis=-1)
        image = tf.tile(image, [1, 1, 3])
    
    # Decode
    class_ids = tf.cast(features['image/class_ids'], tf.int64)
    class_ids = tf.sparse_tensor_to_dense(class_ids)
    
    # Reconstruct the mask indices.
    masks_indices = features['image/masks_indices']
    masks_indices = tf.sparse_tensor_to_dense(masks_indices)
    masks_indices = tf.reshape(masks_indices, (-1, 3))

    # Convert the list of mask indices to a mask tensor.
    mask_shape = (features["image/n_objects"], features["image/width"], features["image/height"])
    masks = tf.sparse_to_dense(masks_indices, output_shape=mask_shape, sparse_values=1)

    labels_dict = {"masks": masks, "class_ids": class_ids, "image_id": features['image/id']}
    
    return image, labels_dict


def build_dataset(tfrecord_path, batch_size, num_epochs=None, shuffle=True, functions=None):
    """Reads input data num_epochs times.
    
    # Parameters
    tfrecord_path : str or Path
        Path of the TFRecord file.
    batch_size : int
        Number of examples per returned batch.
    num_epochs : int
        Number of times to read the input data, or None to train forever.
    shuffle : bool
        Shuffle data or not. Use during training, disable during evaluation.
    functions : list
        A list of Python functions to apply to the dataset.

    # Returns
        A tuple (features, labels) where each element is a map.
    """

    with tf.name_scope('input'):
        # TFRecordDataset opens a binary file and reads one record at a time.
        # `filename` could also be a list of filenames, which will be read in order.
        dataset = tf.data.TFRecordDataset(str(tfrecord_path))

        # The map transformation takes a function and applies it to every element
        # of the dataset.
        dataset = dataset.map(decode_tfrecord)
        
        if functions:
            for func in functions:
                dataset = dataset.map(func)

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        if shuffle:
            dataset = dataset.shuffle(1000 + 3 * batch_size)

        dataset = dataset.repeat(num_epochs)
        
        # When batching with pad the data if needed with the `0` value.
        padded_shapes = ([None, None, None],
                         {"class_ids": [None],
                          "masks": [None, None, None],
                          "image_id": []})
        dataset = dataset.padded_batch(batch_size, padded_shapes=padded_shapes)

    return dataset


def get_data(tfrecord_path, n, shuffle=True, functions=None):
    """Get the data as Numpy array from a TFRecord file.
    
    # Parameters
    tfrecord_path : str or Path
        Path of the TFRecord file.
    n : int
        How many datum to get.
    shuffle : bool
        Shuffle the dataset or not.
    functions : list
        A list of Python functions to apply to the dataset.

    # Returns
        A tuple (features, labels) where each element is a map.
    """
    
    dataset = build_dataset(tfrecord_path, batch_size=n, num_epochs=1, shuffle=shuffle, functions=functions)
    iterator = dataset.make_one_shot_iterator()

    if not tf.executing_eagerly():
        with tf.Session() as sess:
            return sess.run(iterator.get_next())
    else:
        images, annotations = iterator.get_next()
        labels_dict = {"masks": annotations["masks"].numpy(),
                       "class_ids": annotations["class_ids"].numpy(),
                       "image_id": annotations["image_id"].numpy()}
        return images.numpy(), labels_dict