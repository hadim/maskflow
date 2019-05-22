import tensorflow as tf


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


def pad_first_dimension(tensor, num_elements, padded_value):

    # Pad the first dimension.
    tensor_len = tf.shape(tensor)[0]
    paddings = [[0, tf.maximum(num_elements - tensor_len, 0)]]

    # Fill other dimensions with 0 padding.
    num_dim = tf.size(tf.shape(tensor))
    paddings = tf.concat([paddings, tf.tile([[0, 0]], [num_dim - 1, 1])], axis=0)

    return tf.pad(tensor, paddings, constant_values=padded_value)
