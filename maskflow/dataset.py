import tensorflow as tf
import numpy as np

from maskflow import imaging


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def build_features_dict(image, image_id, filename, image_format=None,
                        bboxes=None, masks=None, label_ids=None,
                        label_names=None, masks_format="png"):
    """Build a feature dict for object detection.

    Args:
        image: Array of shape (W, H, C).
        image_id: `int` or `str`, to identify the image.
        filename: `str`, the name of the image file.
        image_format: `str`, image format for encoding, can be `jpeg` or `png`.
        bboxes: `float` array of shape (n_object, x, y, width, height).
        masks: `int` array of shape (n_object, width, height).
        label_ids: `int` array of shape (n_object,).
        label_names: `str` array of shape (n_object,).
        masks_format: `str`, mask format for encoding, can be `jpeg` or `png`.

    Returns:
        A dict of the features.
    """

    # Add channel dimension if needed.
    if len(image.shape) == 3:
        pass
    elif len(image.shape) == 2:
        image = np.expand_dims(image, -1)
    else:
        raise Exception(f"Wrong image shape: {image.shape}")

    # Get image shape.
    image_width, image_height, image_channel = image.shape

    # Encode image.
    image_encoded = imaging.encode_image(image, image_format)

    # Create te feature dict.
    feature_dict = {}

    # Image features
    feature_dict['image_height'] = int64_feature(image_height)
    feature_dict['image_width'] = int64_feature(image_width)
    feature_dict['image_channel'] = int64_feature(image_channel)
    feature_dict['image_filename'] = bytes_feature(filename.encode('utf8'))
    feature_dict['image_id'] = bytes_feature(str(image_id).encode('utf8'))
    feature_dict['image_encoded'] = bytes_feature(image_encoded.numpy())
    feature_dict['image_format'] = bytes_feature(image_format.encode('utf8'))

    # Object features
    if bboxes:
        bboxes_x = bboxes[:, 0]
        bboxes_y = bboxes[:, 1]
        bboxes_width = bboxes[:, 2]
        bboxes_height = bboxes[:, 3]
        feature_dict['bboxes_x'] = float_list_feature(bboxes_x)
        feature_dict['bboxes_y'] = float_list_feature(bboxes_y)
        feature_dict['bboxes_width'] = float_list_feature(bboxes_width)
        feature_dict['bboxes_height'] = float_list_feature(bboxes_height)

    if label_ids:
        feature_dict['label_ids'] = int64_list_feature(label_ids)

    if label_names:
        feature_dict['label_names'] = bytes_list_feature(label_names)

    if masks:
        # Encode masks.
        masks_encoded = []
        for mask in masks:
            mask = image = np.expand_dims(mask, -1)
            mask_encoded = imaging.encode_image(mask, masks_format)
            masks_encoded.append(mask_encoded.numpy())

        feature_dict['masks_encoded'] = bytes_list_feature(masks_encoded)
        feature_dict['masks_format'] = bytes_feature(masks_format.encode("utf8"))

    return feature_dict


def get_feature_description(with_bboxes=True, with_label_names=True,
                            with_label_ids=True, with_masks=True):
    feature_description = {}

    # Image features
    feature_description['image_height'] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['image_width'] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['image_channel'] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['image_filename'] = tf.io.FixedLenFeature([], tf.string)
    feature_description['image_id'] = tf.io.FixedLenFeature([], tf.string)
    feature_description['image_encoded'] = tf.io.FixedLenFeature([], tf.string)
    feature_description['image_format'] = tf.io.FixedLenFeature([], tf.string)

    # Object features
    if with_bboxes:
        feature_description['bboxes_x'] = tf.io.VarLenFeature(tf.float32)
        feature_description['bboxes_y'] = tf.io.VarLenFeature(tf.float32)
        feature_description['bboxes_width'] = tf.io.VarLenFeature(tf.float32)
        feature_description['bboxes_height'] = tf.io.VarLenFeature(tf.float32)

    if with_label_names:
        feature_description['label_names'] = tf.io.VarLenFeature(tf.string)

    if with_label_ids:
        feature_description['label_ids'] = tf.io.VarLenFeature(tf.int64)

    if with_masks:
        feature_description['masks_encoded'] = tf.io.VarLenFeature(tf.string)
        feature_description['masks_format'] = tf.io.FixedLenFeature([], tf.string)

    return feature_description


def _parse_tfrecord(with_bboxes=True, with_label_names=True,
                    with_label_ids=True, with_masks=True):

    # Get feature description
    feature_description = get_feature_description(with_bboxes, with_label_names,
                                                  with_label_ids, with_masks)

    def _fn(datum):
        # Parse the file.
        datum = tf.io.parse_single_example(datum, feature_description)
        # Variable length feature are parser as sparse tensor.
        # So we convert them to dense.
        if with_bboxes:
            datum['bboxes_x'] = datum['bboxes_x'].values
            datum['bboxes_y'] = datum['bboxes_y'].values
            datum['bboxes_width'] = datum['bboxes_width'].values
            datum['bboxes_height'] = datum['bboxes_height'].values

        if with_label_names:
            datum['label_names'] = datum['label_names'].values

        if with_label_ids:
            datum['label_ids'] = datum['label_ids'].values

        if with_masks:
            datum['masks_encoded'] = datum['masks_encoded'].values
        return datum

    return _fn


def _decode_image(datum):
    # Convert image to tensor.
    encoded_image = datum['image_encoded']
    datum['image'] = tf.image.decode_image(encoded_image)
    return datum


def _decode_masks(datum):
    """Convert list of masks to tensor.
    """
    encoded_masks = datum['masks_encoded']
    datum['masks'] = tf.map_fn(lambda x: tf.image.decode_image(x, channels=1),
                               encoded_masks, dtype=tf.uint8)
    # Remove the channel dimension (always equal to 1).
    datum['masks'] = datum['masks'][:, :, :, 0]
    return datum


def parse(tfrecord_path, with_bboxes=True, with_label_names=True,
          with_label_ids=True, with_masks=True):
    """Parse a Maskflow TFRecord file.

    Args:
        tfrecord_path: `str`, path to TFRecord file.
        with_bboxes: `bool`, parse bounding boxes.
        with_label_names: `bool`, parse label names.
        with_label_ids: `bool`, parse label indices.
        with_masks: `bool`, parse masks.

    Returns:
        A Tensorflow dataset.
    """

    dataset = tf.data.TFRecordDataset(str(tfrecord_path))

    _feature_parser_fn = _parse_tfrecord(with_bboxes, with_label_names,
                                         with_label_ids, with_masks)
    dataset = dataset.map(_feature_parser_fn)
    dataset = dataset.map(_decode_image)
    dataset = dataset.map(_decode_masks)
    return dataset
