import tempfile

import numpy as np
import tensorflow as tf

import maskflow


def make_fake_object():
    image_id = 874
    filename = "fake_image.png"
    class_names = ["object_1", "object_2", "object_3"]
    encoded_mask_format = "png"

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
                                                        label_ids, class_names, encoded_mask_format=encoded_mask_format)

    excepted_keys = ["image_height", "image_width", "image_channel", "image_filename",
                     "image_id", "image_encoded", "image_format", "bboxes_x",
                     "bboxes_y", "bboxes_width", "bboxes_height", "label_names",
                     "label_ids", "masks_encoded", "masks_format"]

    assert set(features_dict.keys()) == set(excepted_keys)

    masks_feature = features_dict["image_encoded"]
    assert masks_feature.bytes_list.value[0][:10] == b'\x89PNG\r\n\x1a\n\x00\x00'


def test_parse_tfrecord():
    objects = make_fake_object()
    image, image_id, filename, masks, label_ids, class_names, encoded_mask_format = objects

    features_dict = maskflow.dataset.build_features_dict(image, image_id, filename, masks,
                                                        label_ids, class_names, encoded_mask_format=encoded_mask_format)

    _, tfrecord_path = tempfile.mkstemp()
    example = tf.train.Example(features=tf.train.Features(feature=features_dict))

    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        writer.write(example.SerializeToString())

    dataset = maskflow.dataset.parse(tfrecord_path)

    datum = next(iter(dataset))
    excepted_keys = ["image_height", "image_width", "image_channel", "image_filename",
                     "image_id", "image_encoded", "image_format", "bboxes_x",
                     "bboxes_y", "bboxes_width", "bboxes_height", "label_names",
                     "label_ids", "masks_encoded", "masks_format", 'image', 'masks']

    assert datum["image_encoded"].numpy()[0][:10] == b'\x89PNG\r\n\x1a\n\x00\x00'
