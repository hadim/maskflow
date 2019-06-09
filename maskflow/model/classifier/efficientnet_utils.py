import tensorflow as tf


def drop_connect(inputs, is_training, drop_connect_rate):
  """Apply drop connect."""
  if not is_training:
    return inputs

  # Compute keep_prob
  # TODO(tanmingxing): add support for training progress.
  keep_prob = 1.0 - drop_connect_rate

  # Compute drop_connect tensor
  batch_size = tf.shape(inputs)[0]
  random_tensor = keep_prob
  random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  output = (inputs / keep_prob) * binary_tensor
  return output


class Swish(tf.keras.layers.Layer):
  """Swish stands for self-gated activation function.
  This is new activation function defined by Google.
  Like ReLU, Swish is unbounded above and bounded below, below is the paper.
  https://arxiv.org/pdf/1710.05941v1.pdf
  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.
  Output shape:
      Same shape as the input.
  Arguments:
      beta: float >= 0. Scaling factor
      trainable: whether to learn the scaling factor during training or not
  """

  def __init__(self, beta=1.0, trainable=False, **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True
    self.beta = beta
    self.trainable = trainable
    self.scaling_factor = None

  def build(self, input_shape):
    self.scaling_factor = tf.constant(self.beta,
                                      dtype=tf.float32,
                                      name='scaling_factor')
    if self.trainable:
      self.trainable_weights.append(self.scaling_factor)
    super().build(input_shape)

  def call(self, inputs, **kwargs):
    return inputs * tf.sigmoid(self.scaling_factor * inputs)

  def get_config(self):
    config = {
        'beta': self.get_weights()[0] if self.trainable else self.beta,
        'trainable': self.trainable
    }
    base_config = super(Swish, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  # pylint: disable=arguments-differ
  def compute_output_signature(self, input_shape):
    return input_shape


# class TpuBatchNormalization(tf.layers.BatchNormalization):
#   # class TpuBatchNormalization(tf.layers.BatchNormalization):
#   """Cross replica batch normalization."""

#   def __init__(self, fused=False, **kwargs):
#     if fused in (True, None):
#       raise ValueError('TpuBatchNormalization does not support fused=True.')
#     super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

#   def _cross_replica_average(self, t, num_shards_per_group):
#     """Calculates the average value of input tensor across TPU replicas."""
#     num_shards = tpu_function.get_tpu_context().number_of_shards
#     group_assignment = None
#     if num_shards_per_group > 1:
#       if num_shards % num_shards_per_group != 0:
#         raise ValueError('num_shards: %d mod shards_per_group: %d, should be 0'
#                          % (num_shards, num_shards_per_group))
#       num_groups = num_shards // num_shards_per_group
#       group_assignment = [[
#           x for x in range(num_shards) if x // num_shards_per_group == y
#       ] for y in range(num_groups)]
#     return tpu_ops.cross_replica_sum(t, group_assignment) / tf.cast(
#         num_shards_per_group, t.dtype)

#   def _moments(self, inputs, reduction_axes, keep_dims):
#     """Compute the mean and variance: it overrides the original _moments."""
#     shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
#         inputs, reduction_axes, keep_dims=keep_dims)

#     num_shards = tpu_function.get_tpu_context().number_of_shards or 1
#     if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
#       num_shards_per_group = 1
#     else:
#       num_shards_per_group = max(8, num_shards // 4)
#     tf.logging.info('TpuBatchNormalization with num_shards_per_group %s',
#                     num_shards_per_group)
#     if num_shards_per_group > 1:
#       # Each group has multiple replicas: here we compute group mean/variance by
#       # aggregating per-replica mean/variance.
#       group_mean = self._cross_replica_average(shard_mean, num_shards_per_group)
#       group_variance = self._cross_replica_average(shard_variance,
#                                                    num_shards_per_group)

#       # Group variance needs to also include the difference between shard_mean
#       # and group_mean.
#       mean_distance = tf.square(group_mean - shard_mean)
#       group_variance += self._cross_replica_average(mean_distance,
#                                                     num_shards_per_group)
#       return (group_mean, group_variance)
#     else:
#       return (shard_mean, shard_variance)
