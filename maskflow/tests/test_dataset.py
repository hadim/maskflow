import tempfile

import numpy as np
import tensorflow as tf

import maskflow


def make_fake_object():
    image_id = 874
    filename = "fake_image.png"
    class_names = ["object_1", "object_2", "object_3"]
    encoded_mask_format = "tiff"

    image = np.zeros((128, 128), dtype="uint8")
    image[20:40, 55:95] = 255

    n_objects = 25
    masks = np.zeros((n_objects, 128, 128), dtype="uint8")
    masks[:, 20:40, 55:95] = 255
    label_ids = np.random.choice([0, 1, 2], n_objects)

    return image, image_id, filename, masks, label_ids, class_names, encoded_mask_format


def test_build_feature_dict():
    objects = make_fake_object()
    image, image_id, filename, masks, label_ids, class_names, encoded_mask_format = objects

    features_dict = maskflow.dataset.build_features_dict(image, image_id, filename, masks,
                                                        label_ids, class_names, encoded_mask_format)

    excepted_keys = ['image/height', 'image/width', 'image/channel', 'image/filename',
                     'image/source_id', 'image/encoded', 'image/key/sha256', 'image/format',
                     'image/object/bbox/x', 'image/object/bbox/y', 'image/object/bbox/width',
                     'image/object/bbox/height', 'image/object/class/text',
                     'image/object/class/label', 'image/object/masks', 'image/object/mask_format']

    assert set(features_dict.keys()) == set(excepted_keys)

    sha256_feature = features_dict["image/key/sha256"]
    assert sha256_feature.bytes_list.value[0] == b'13b6ea6e31c7d27c8545fdba4ed8c23cfdd6a011efde0831292d985306ecaadb'

    masks_feature = features_dict["image/object/masks"]
    assert masks_feature.bytes_list.value[0][:10] == b'SUkqAAgAAA'


def test_parse_tfrecord():
    objects = make_fake_object()
    image, image_id, filename, masks, label_ids, class_names, encoded_mask_format = objects

    features_dict = maskflow.dataset.build_features_dict(image, image_id, filename, masks,
                                                        label_ids, class_names, encoded_mask_format)

    _, tfrecord_path = tempfile.mkstemp()
    example = tf.train.Example(features=tf.train.Features(feature=features_dict))

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        writer.write(example.SerializeToString())

    dataset = maskflow.dataset.parse_tfrecord(tfrecord_path)

    datum = next(iter(dataset))
    excepted_keys = ['image/height', 'image/width', 'image/channel', 'image/filename',
                     'image/source_id', 'image/encoded', 'image/key/sha256', 'image/format',
                     'image/object/bbox/x', 'image/object/bbox/y', 'image/object/bbox/width',
                     'image/object/bbox/height', 'image/object/class/text',
                     'image/object/class/label', 'image/object/masks', 'image/object/mask_format']

    assert set(datum.keys()) == set(excepted_keys)

    assert datum["image/key/sha256"].numpy() == b'13b6ea6e31c7d27c8545fdba4ed8c23cfdd6a011efde0831292d985306ecaadb'

    assert datum["image/object/masks"].numpy()[0][:10] == b'SUkqAAgAAA'
