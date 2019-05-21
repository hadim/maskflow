import hashlib

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


def build_features_dict(image, image_id, filename, masks,
                        label_ids, class_names,
                        encoded_mask_format="tiff"):

    # Get image informations
    if len(image.shape) == 3:
        image_width, image_height, image_channel = image.shape
    elif len(image.shape) == 2:
        image_width, image_height = image.shape
        image_channel = 1
    else:
        raise Exception(f"Wrong image shape: {image.shape}")

    image_format = filename.split(".")[-1]
    encoded_image = imaging.encode_image(image, image_format)
    sha256key = hashlib.sha256(encoded_image.encode("utf8")).hexdigest()

    # Get a list of class from the object label ids.
    label_names = [class_names[class_id].encode("utf8") for class_id in label_ids]

    # Find bounding boxes coordinates from mask and encode the masks.

    bboxes_x = []
    bboxes_y = []
    bboxes_width = []
    bboxes_height = []
    encoded_masks = []

    for i, mask in enumerate(masks):
        coords = np.argwhere(mask > 0)
        if coords.shape[0] == 0:
          raise Exception(f"Empty mask at position {i}: {coords.shape}")

        xmin = coords[:, 0].min()
        xmax = coords[:, 0].max()
        ymin = coords[:, 1].min()
        ymax = coords[:, 1].max()

        bboxes_x.append(xmin)
        bboxes_y.append(ymin)
        bboxes_width.append(xmax - xmin)
        bboxes_height.append(ymax - ymin)

        encoded_mask = imaging.encode_image(mask, encoded_mask_format)
        encoded_masks.append(encoded_mask.encode("utf8"))

    feature_dict = {}

    # Image features
    feature_dict['image/height'] = int64_feature(image_height)
    feature_dict['image/width'] = int64_feature(image_width)
    feature_dict['image/channel'] = int64_feature(image_channel)
    feature_dict['image/filename'] = bytes_feature(filename.encode('utf8'))
    feature_dict['image/source_id'] = bytes_feature(str(image_id).encode('utf8'))
    feature_dict['image/encoded'] = bytes_feature(encoded_image.encode('utf8'))
    feature_dict['image/key/sha256'] = bytes_feature(sha256key.encode('utf8'))
    feature_dict['image/format'] = bytes_feature(image_format.encode('utf8'))

    # Object features
    feature_dict['image/object/bbox/x'] = float_list_feature(bboxes_x)
    feature_dict['image/object/bbox/y'] = float_list_feature(bboxes_y)
    feature_dict['image/object/bbox/width'] = float_list_feature(bboxes_width)
    feature_dict['image/object/bbox/height'] = float_list_feature(bboxes_height)
    feature_dict['image/object/class/text'] = bytes_list_feature(label_names)
    feature_dict['image/object/class/label'] = int64_list_feature(label_ids)
    feature_dict['image/object/masks'] = bytes_list_feature(encoded_masks)
    feature_dict['image/object/mask_format'] = bytes_feature(encoded_mask_format.encode("utf8"))

    # Features currently not used
    #feature_dict['image/object/is_crowd'] = int64_list_feature(is_crowd)
    #feature_dict['image/object/area'] = float_list_feature(area)
    #feature_dict['image/caption'] = bytes_list_feature(captions)

    return feature_dict


def parse_tfrecord(tfrecord_path):
    feature_description = {}

    # Image features
    feature_description['image/height'] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['image/width'] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['image/channel'] = tf.io.FixedLenFeature([], tf.int64)
    feature_description['image/filename'] = tf.io.FixedLenFeature([], tf.string)
    feature_description['image/source_id'] = tf.io.FixedLenFeature([], tf.string)
    feature_description['image/encoded'] = tf.io.FixedLenFeature([], tf.string)
    feature_description['image/key/sha256'] = tf.io.FixedLenFeature([], tf.string)
    feature_description['image/format'] = tf.io.FixedLenFeature([], tf.string)

    # Object features
    feature_description['image/object/bbox/x'] = tf.io.VarLenFeature(tf.float32)
    feature_description['image/object/bbox/y'] = tf.io.VarLenFeature(tf.float32)
    feature_description['image/object/bbox/width'] = tf.io.VarLenFeature(tf.float32)
    feature_description['image/object/bbox/height'] = tf.io.VarLenFeature(tf.float32)
    feature_description['image/object/class/text'] = tf.io.VarLenFeature(tf.string)
    feature_description['image/object/class/label'] = tf.io.VarLenFeature(tf.int64)
    feature_description['image/object/masks'] = tf.io.VarLenFeature(tf.string)
    feature_description['image/object/mask_format'] = tf.io.FixedLenFeature([], tf.string)

    def _parse_example(example):
        datum = tf.io.parse_single_example(example, feature_description)

        # Variable length feature are parser as sparse tensor.
        # So we convert them to dense.
        datum['image/object/bbox/x'] = datum['image/object/bbox/x'].values
        datum['image/object/bbox/y'] = datum['image/object/bbox/y'].values
        datum['image/object/bbox/width'] = datum['image/object/bbox/width'].values
        datum['image/object/bbox/height'] = datum['image/object/bbox/height'].values
        datum['image/object/class/text'] = datum['image/object/class/text'].values
        datum['image/object/class/label'] = datum['image/object/class/label'].values
        datum['image/object/masks'] = datum['image/object/masks'].values

        return datum

    train_dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    return train_dataset.map(_parse_example)
