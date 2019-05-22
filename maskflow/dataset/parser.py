import tensorflow as tf

from . import get_feature_description
from . import preprocess_dataset


def _parse_tfrecord(with_bboxes=True, with_label_names=True,
                    with_label_ids=True, with_masks=True):

    # Get feature description
    feature_description = get_feature_description(with_bboxes, with_label_names,
                                                  with_label_ids, with_masks)

    def _fn(feature):
        # Parse the file.
        feature = tf.io.parse_single_example(feature, feature_description)
        # Variable length feature are parser as sparse tensor.
        # So we convert them to dense.
        if with_bboxes:
            feature['bboxes_x'] = feature['bboxes_x'].values
            feature['bboxes_y'] = feature['bboxes_y'].values
            feature['bboxes_width'] = feature['bboxes_width'].values
            feature['bboxes_height'] = feature['bboxes_height'].values

        if with_label_names:
            feature['label_names'] = feature['label_names'].values

        if with_label_ids:
            feature['label_ids'] = feature['label_ids'].values

        if with_masks:
            feature['masks_encoded'] = feature['masks_encoded'].values
        return feature

    return _fn


def _decode_image(feature):
    # Convert image to tensor.
    encoded_image = feature['image_encoded']
    feature['image'] = tf.image.decode_image(encoded_image)
    return feature


def _decode_bounding_boxes(feature):
    """Convert bboxes to (x, y, w, h).
    """
    feature['bboxes'] = tf.stack([feature['bboxes_x'],
                                  feature['bboxes_y'],
                                  feature['bboxes_width'],
                                  feature['bboxes_height']],
                                  axis=1)
    return feature


def _decode_masks(feature):
    """Convert list of masks to tensor.
    """
    # TODO: the current function fails when there is no mask in
    # feature['masks_encoded']. It should work on empty an tensor.
    # See TODO SNIPPET TO REPRODUCE
    encoded_masks = feature['masks_encoded']
    feature['masks'] = tf.map_fn(lambda x: tf.image.decode_image(x, channels=1),
                                 encoded_masks, dtype=tf.uint8)
    # Remove the channel dimension (always equal to 1).
    feature['masks'] = feature['masks'][:, :, :, 0]
    return feature


def _prune_features(feature):
    keys_to_prune = ["image_encoded", "image_format", "bboxes_x",
                     "bboxes_y", "bboxes_width", "bboxes_height",
                     "masks_encoded", "masks_format"]
    for key in keys_to_prune:
        if key in feature.keys():
            feature.pop(key)
    return feature


def parse(tfrecord_path, config, do_preprocess=True,
          with_bboxes=True, with_label_names=True,
          with_label_ids=True, with_masks=True):
    """Parse a Maskflow TFRecord file.

    Args:
        tfrecord_path: `str`, path to TFRecord file.
        config: `dict`, Maskflow dataset and model parameters.
        do_preprocess: `bool`, whether to preprocess the dataset.
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

    if do_preprocess:
        dataset = preprocess_dataset(dataset, config['DATASET']['MAX_NUM_INSTANCES'])

    return dataset
