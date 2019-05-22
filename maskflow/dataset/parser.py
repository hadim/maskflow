import tensorflow as tf

from . import get_feature_description


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


def _decode_bounding_boxes(datum):
    """Convert bboxes to (x, y, w, h).
    """
    datum['bboxes'] = tf.stack([datum['bboxes_x'],
                                datum['bboxes_y'],
                                datum['bboxes_width'],
                                datum['bboxes_height']],
                                axis=1)
    return datum


def _decode_masks(datum):
    """Convert list of masks to tensor.
    """
    # TODO: the current function fails when there is no mask in
    # datum['masks_encoded']. It should work on empty an tensor.
    # See TODO SNIPPET TO REPRODUCE
    encoded_masks = datum['masks_encoded']
    datum['masks'] = tf.map_fn(lambda x: tf.image.decode_image(x, channels=1),
                               encoded_masks, dtype=tf.uint8)
    # Remove the channel dimension (always equal to 1).
    datum['masks'] = datum['masks'][:, :, :, 0]
    return datum


def _prune_features(datum):
    keys_to_prune = ["image_encoded", "image_format", "bboxes_x",
                     "bboxes_y", "bboxes_width", "bboxes_height",
                     "masks_encoded", "masks_format"]
    for key in keys_to_prune:
        if key in datum.keys():
            datum.pop(key)
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

    # Parse TFRecord file.
    _feature_parser_fn = _parse_tfrecord(with_bboxes, with_label_names,
                                         with_label_ids, with_masks)
    dataset = dataset.map(_feature_parser_fn)

    # Convert `image_encoded` to a Tensor.
    dataset = dataset.map(_decode_image)

    # Convert `masks_encoded` to a Tensor.
    if with_masks:
        dataset = dataset.map(_decode_masks)

    # Convert bboxes informations to a Tensor of (x, y, w, h).
    dataset = dataset.map(_decode_bounding_boxes)

    # Prune useless features.
    dataset = dataset.map(_prune_features)

    return dataset


def preprocess(dataset, batch=1):
    """Preprocess a Maskflow dataset.
    """

    return dataset
