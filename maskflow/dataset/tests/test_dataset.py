import tempfile

import numpy as np
import tensorflow as tf

import maskflow


def make_fake_object():
  image = np.zeros((128, 128), dtype="uint8")
  image[20:40, 55:95] = 255

  n_objects = 25
  masks = np.zeros((n_objects, 128, 128), dtype="uint8")
  masks[:, 20:40, 55:95] = 255

  label_ids = np.random.choice([0, 1, 2], n_objects)

  # class_names = ["object_1", "object_2", "object_3"]
  # label_names = [class_names[class_id].encode("utf8") for class_id in label_ids]

  build_features_args = {}
  build_features_args['image'] = image
  build_features_args['image_id'] = 874
  build_features_args['filename'] = "fake_image.png"
  build_features_args['image_format'] = "png"
  build_features_args['bboxes'] = maskflow.bbox.from_masks(masks)
  build_features_args['masks'] = masks
  build_features_args['label_ids'] = label_ids
  build_features_args['masks_format'] = "png"

  return build_features_args


def test_build_feature_dict():
  build_features_args = make_fake_object()

  features_dict = maskflow.dataset.build_features_dict(**build_features_args)

  excepted_keys = ['image_filename', 'bboxes_width', 'bboxes_height', 'label_ids',
                   'image_width', 'image_height', 'image_id', 'image_format',
                   'masks_encoded', 'image_channel', 'masks_format', 'image_encoded',
                   'bboxes_x', 'bboxes_y']

  print(set(features_dict.keys()))
  assert set(features_dict.keys()) == set(excepted_keys)

  masks_feature = features_dict["image_encoded"]
  assert masks_feature.bytes_list.value[0][:10] == b'\x89PNG\r\n\x1a\n\x00\x00'


def test_parse_tfrecord():
  config = maskflow.get_default_config()

  build_features_args = make_fake_object()

  features_dict = maskflow.dataset.build_features_dict(**build_features_args)

  _, tfrecord_path = tempfile.mkstemp()
  example = tf.train.Example(features=tf.train.Features(feature=features_dict))

  with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
    writer.write(example.SerializeToString())

  dataset = maskflow.dataset.parse(tfrecord_path, config)

  feature = next(iter(dataset))
  excepted_keys = ['image_filename', 'bboxes', 'image_width', 'image_height',
                   'image_id', 'masks', 'label_names', 'image_channel', 'image',
                   'label_ids']

  print(set(feature.keys()))
  assert set(feature.keys()) == set(excepted_keys)

  print(feature["masks"].numpy().shape)
  assert feature["masks"].numpy().shape == (200, 128, 128)

  print(feature["image"].numpy().shape)
  assert feature["image"].numpy().shape == (128, 128, 1)
