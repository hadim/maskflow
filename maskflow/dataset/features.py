import tensorflow as tf
import numpy as np

from .utils import int64_feature
from .utils import bytes_feature
from .utils import float_list_feature
from .utils import int64_list_feature
from .utils import bytes_list_feature

from .. import imaging


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


# pylint: disable=too-many-locals
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
  if bboxes is not None:
    if bboxes.shape[0] > 0:
      bboxes_x = bboxes[:, 0]
      bboxes_y = bboxes[:, 1]
      bboxes_width = bboxes[:, 2]
      bboxes_height = bboxes[:, 3]
    else:
      bboxes_x = []
      bboxes_y = []
      bboxes_width = []
      bboxes_height = []

    feature_dict['bboxes_x'] = float_list_feature(bboxes_x)
    feature_dict['bboxes_y'] = float_list_feature(bboxes_y)
    feature_dict['bboxes_width'] = float_list_feature(bboxes_width)
    feature_dict['bboxes_height'] = float_list_feature(bboxes_height)

  if label_ids is not None:
    feature_dict['label_ids'] = int64_list_feature(label_ids)

  if label_names is not None:
    feature_dict['label_names'] = bytes_list_feature(label_names)

  if masks is not None:
    # Encode masks.
    masks_encoded = []
    for mask in masks:
      mask = image = np.expand_dims(mask, -1)
      mask_encoded = imaging.encode_image(mask, masks_format)
      masks_encoded.append(mask_encoded.numpy())

    feature_dict['masks_encoded'] = bytes_list_feature(masks_encoded)
    feature_dict['masks_format'] = bytes_feature(masks_format.encode("utf8"))

  return feature_dict
