# Freely inspired from
# https://github.com/tensorflow/tpu/blob/b26a9244d326136c565968748d4475d366d1287d/models/official/resnet/resnet_model.py

import tensorflow as tf
import tensorflow.keras.layers as layers


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-4


class BatchNormRelu(tf.keras.layers.Layer):
    """Do Batch Norm. -> ReLu.
    """

    def __init__(self, init_zero=False, do_relu=True, name='', data_format='channels_last'):

        super().__init__(name=name)

        if init_zero:
            gamma_initializer = tf.zeros_initializer()
        else:
            gamma_initializer = tf.ones_initializer()

        if data_format == 'channels_first':
            axis = 1
        else:
            axis = 3

        self.bn = layers.BatchNormalization(axis=axis,
                                            momentum=_BATCH_NORM_DECAY,
                                            epsilon=_BATCH_NORM_EPSILON,
                                            fused=True,
                                            center=True,
                                            scale=True,
                                            gamma_initializer=gamma_initializer)

        self.relu = None
        if do_relu:
            self.relu = tf.nn.relu

    def call(self, input_tensor, training=False):
        x = self.bn(input_tensor, training=training)

        if self.relu:
            x = self.relu(x)

        return x


class ConvNormReLuPool(tf.keras.layers.Layer):
    """Do Convolution -> Batch Norm. -> ReLu -> Pooling.
    """

    def __init__(self, filters, kernel_size, strides, padding='same',
                 init_zero=False, do_relu=True, pool_size=3,
                 pool_strides=2, pool_padding='same', name='',
                 data_format='channels_last'):

        super().__init__(name=name)
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        self.bn_relu = BatchNormRelu(init_zero=init_zero, do_relu=do_relu, data_format=data_format)
        self.pool = layers.MaxPooling2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding)


    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn_relu(x, training=training)
        x = self.pool(x)
        return x


class ClassifyBlock(tf.keras.layers.Layer):
    """Do Average Pooling -> FC -> Activation (softmax).
    """

    def __init__(self, num_classes, fc_length=512, activation='softmax', name='', data_format='channels_last'):

        super().__init__(name=name)

        kernel_initializer = tf.random_normal_initializer(stddev=0.01)
        #kernel_initializer = 'glorot_uniform'

        self.fc_length = fc_length
        self.average_pool = layers.AveragePooling2D(pool_size=7, strides=1, padding='valid', data_format=data_format)
        self.dense = layers.Dense(units=num_classes, activation=activation, kernel_initializer=kernel_initializer)

    def call(self, input_tensor, training=False):
        x = self.average_pool(input_tensor)
        x = tf.reshape(x, [-1, self.fc_length])
        x = self.dense(x)
        return x


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides, use_projection=False, data_format='channels_last', name=''):
        super().__init__(name=name)

        self.bn_relu_1 = BatchNormRelu(init_zero=False, do_relu=True, data_format=data_format)

        self.projection = None
        if use_projection:
            # Projection shortcut in first layer to match filters and strides
            self.projection = layers.Conv2D(filters=filters, kernel_size=1, strides=strides, padding='same')

        self.conv_1 = layers.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same')

        self.bn_relu_2 = BatchNormRelu(init_zero=False, do_relu=True, data_format=data_format)
        self.conv_2 = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')

    def call(self, input_tensor, training=False):

        shortcut = input_tensor
        x = self.bn_relu_1(input_tensor, training=training)

        if self.projection:
            shortcut = self.projection(x)

        x = self.conv_1(x)
        x = self.bn_relu_2(x, training=training)
        x = self.conv_2(x)

        return x + shortcut


class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides, use_projection=False, data_format='channels_last', name=''):
        super().__init__(name=name)

        self.bn_relu_1 = BatchNormRelu(init_zero=False, do_relu=True, data_format=data_format)

        self.projection = None
        if use_projection:
            # Projection shortcut only in first block within a group. Bottleneck blocks
            # end with 4 times the number of filters.
            filters_out = 4 * filters
            self.projection = layers.Conv2D(filters=filters_out, kernel_size=1, strides=strides, padding='same')

        self.conv_1 = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')

        self.bn_relu_2 = BatchNormRelu(init_zero=False, do_relu=True, data_format=data_format)
        self.conv_2 = layers.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='same')

        self.bn_relu_3 = BatchNormRelu(init_zero=False, do_relu=True, data_format=data_format)
        self.conv_3 = layers.Conv2D(filters=4 * filters, kernel_size=1, strides=1, padding='same')

    def call(self, input_tensor, training=False):

        shortcut = input_tensor
        x = self.bn_relu_1(input_tensor, training=training)

        if self.projection:
            shortcut = self.projection(x)

        x = self.conv_1(x)
        x = self.bn_relu_2(x, training=training)
        x = self.conv_2(x)
        x = self.bn_relu_3(x, training=training)
        x = self.conv_3(x)

        return x + shortcut


class BlockGroup(tf.keras.layers.Layer):
    """One group of blocks for the ResNet model.
    """

    def __init__(self, filters, strides, n_blocks, block_fn, data_format='channels_last', name=''):

        super().__init__(name=name)

        # Only the first block per BlockGroup uses projection shortcut and strides.
        self.first_block = block_fn(filters, strides, use_projection=True, data_format=data_format)

        self.blocks = []
        for _ in range(1, n_blocks):
            block = block_fn(filters, strides=1, use_projection=False, data_format=data_format)
            self.blocks.append(block)

    def call(self, input_tensor, training=False):
        x = self.first_block(input_tensor, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return x


class ResNet(tf.keras.Model):
    """ResNet v2 model for a given size and number of output classes.

    Version 2 means in block group we do `BN -> ReLu -> Conv` instead of
    `Conv -> BN -> ReLu`.

    TODO: Implement DropBlock.

    Args:
        resnet_size: int, size of the model.
        num_classes: int, number of classes (not used if with_head is False).
        include_top: bool, provide the classification layers or not.
        activation: bool, softmax function to use on final classification layer.
        data_format: str, either "channels_first" for `[batch, channels, height,
            width]` or "channels_last for `[batch, height, width, channels]`.
    """

    def __init__(self, size=50, num_classes=10, include_top=True, activation='softmax', data_format='channels_last', **kwargs):

        super().__init__(name=f'ResNet_{size}', **kwargs)

        params = {18: {'block': ResidualBlock, 'layers': [2, 2, 2, 2]},
                  34: {'block': ResidualBlock, 'layers': [3, 4, 6, 3]},
                  50: {'block': BottleneckBlock, 'layers': [3, 4, 6, 3]},
                  101: {'block': BottleneckBlock, 'layers': [3, 4, 23, 3]},
                  152: {'block': BottleneckBlock, 'layers': [3, 8, 36, 3]},
                  200: {'block': BottleneckBlock, 'layers': [3, 24, 36, 3]}}

        if size not in params:
            raise ValueError(f'Not a valid Resnet size: {size}.'
                             f"Please use sizes among {params.keys()}")

        block_fn = params[size]['block']
        n_layers = params[size]['layers']

        self.c1 = ConvNormReLuPool(filters=64, kernel_size=7, strides=2, padding='same', init_zero=False,
                                   do_relu=True, pool_size=3, pool_strides=2, pool_padding='same', name='block_group_1')

        self.c2 = BlockGroup(filters=64, strides=1, n_blocks=n_layers[0], block_fn=block_fn,
                             data_format=data_format, name='block_group_2')

        self.c3 = BlockGroup(filters=128, strides=2, n_blocks=n_layers[1], block_fn=block_fn,
                             data_format=data_format, name='block_group_3')

        self.c4 = BlockGroup(filters=256, strides=2, n_blocks=n_layers[2], block_fn=block_fn,
                             data_format=data_format, name='block_group_4')

        self.c5 = BlockGroup(filters=512, strides=2, n_blocks=n_layers[3], block_fn=block_fn,
                             data_format=data_format, name='block_group_5')

        self.classify = None
        if include_top:
            fc_length = 512 if block_fn == ResidualBlock else 2048
            self.classify = ClassifyBlock(num_classes, fc_length=fc_length, activation=activation,
                                          name='classify_block', data_format=data_format)


    def call(self, input_tensor, training=False):

        x = self.c1(input_tensor, training=training)
        x = self.c2(x, training=training)
        x = self.c3(x, training=training)
        x = self.c4(x, training=training)
        x = self.c5(x, training=training)

        if self.classify:
            x = self.classify(x, training=training)

        return x



class ResNet18(ResNet):
    """See `maskflow.model.ResNet` for details.
    """
    def __init__(self, **kwargs):
        super().__init__(size=18, **kwargs)


class ResNet34(ResNet):
    """See `maskflow.model.ResNet` for details.
    """
    def __init__(self, **kwargs):
        super().__init__(size=34, **kwargs)


class ResNet50(ResNet):
    """See `maskflow.model.ResNet` for details.
    """
    def __init__(self, **kwargs):
        super().__init__(size=50, **kwargs)


class ResNet101(ResNet):
    """See `maskflow.model.ResNet` for details.
    """
    def __init__(self, **kwargs):
        super().__init__(size=101, **kwargs)


class ResNet152(ResNet):
    """See `maskflow.model.ResNet` for details.
    """
    def __init__(self, **kwargs):
        super().__init__(size=152, **kwargs)


class ResNet200(ResNet):
    """See `maskflow.model.ResNet` for details.
    """
    def __init__(self, **kwargs):
        super().__init__(size=200, **kwargs)
