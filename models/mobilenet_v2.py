import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, GlobalAveragePooling2D, Dense
from models.mobilenet import _conv2d_block, _depthwise_separable_block


def _inverted_residual_block(input, expansion_factor, strides, pw_filters, alpha, block_id):
    input_channels = int(input.shape[-1])
    x = input

    # Expansion
    if expansion_factor != 1:
        x = Conv2D(filters=int(expansion_factor * input_channels * alpha),
                kernel_size=1,
                strides=1,
                padding='same',
                use_bias=False,
                name=f'expansion_conv_{block_id}')(x)
        x = BatchNormalization(name=f'expansion_bn_{block_id}')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name=f'expansion_relu6_{block_id}')(x)

    # Depthwise & Pointwise (projection)
    x = _depthwise_separable_block(x, strides, pw_filters, alpha, block_id)

    # Skip connection
    if strides == 1 and input_channels == int(pw_filters * alpha):
        x = tf.keras.layers.add([x, input], name=f'add_{block_id}')

    return x


class MobileNetV2():
    def __init__(self, input_shape=None, n_classes=None, alpha=1.0, dropout=None):
        '''
            Args:
                input_shape: (H, W, C). Default to None
                n_class: No. classes
                alpha: Width multiplier in the original paper (for adjusting no. filter in each layer). Default to 1.0
                dropout: Dropout rate. Default to None
        '''
        if input_shape is None:
            raise ValueError('Input shape must be specified')

        self.input_shape = input_shape
        self.n_classes = n_classes
        self.alpha = alpha
        self.dropout = dropout

        self.model = self.__build_model()

    def get_model(self):
        return self.model

    def __build_model(self):
        input = tf.keras.Input(shape=self.input_shape)

        inverted_residual_stack = [
            # expansion_factor , pw_filters, n, strides
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # First Conv2D layer
        x = _conv2d_block(input, filters=32, kernel_size=3, strides=2, alpha=self.alpha, block_id=1)

        # Inverted residual blocks
        block_id = 1
        for expansion_factor, pw_filters, n, strides in inverted_residual_stack:
            for i in range(n):
                if i != 0:
                    strides = 1
                x = _inverted_residual_block(x, expansion_factor, strides, pw_filters, self.alpha, block_id)
                block_id += 1

        # Last Conv2D layer
        x = Conv2D(filters=int(1280 * self.alpha),
                        kernel_size=1,
                        strides=1,
                        padding='same',
                        use_bias=False,
                        name='conv_last')(x)
        x = BatchNormalization(name='bn_last')(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name='relu6_last')(x)

        # Global Average Pooling
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        if self.dropout is not None:
            x = tf.keras.layers.Dropout(rate=self.dropout, name="dropout")(x)

        # Output layer
        x = Dense(units=self.n_classes, activation='softmax', name='output')(x)

        return tf.keras.Model(inputs=input, outputs=x)

