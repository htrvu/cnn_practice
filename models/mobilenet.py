import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, GlobalAveragePooling2D, Dense, ReLU, ZeroPadding2D

def _conv2d_block(input, filters, kernel_size, strides, alpha, batch_norm, block_id):
    x = Conv2D(filters=int(filters * alpha),
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               use_bias=False,
               name=f'conv_{block_id}')(input)
    if batch_norm:
        x = BatchNormalization(name=f'bn_{block_id}')(x)
    x = ReLU(max_value=6.0, name=f'relu6_{block_id}')(x)
    return x


def _depthwise_separable_block(input, strides, pw_filters, alpha, batch_norm, block_id):
    x = input

    # just a little demo (ref from tensorflow implementation)
    if strides != 1:
        x = ZeroPadding2D(((0, 1), (0, 1)), name="depthwise_pad_%d" % block_id)(input)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3,
                        strides=strides,
                        padding="same" if strides == 1 else "valid",
                        use_bias=False,
                        name=f'depthwise_cond_{block_id}')(x)
    if batch_norm:
        x = BatchNormalization(name=f'depthwise_bn_{block_id}')(x)
    x = ReLU(max_value=6.0, name=f'depthwise_relu6_{block_id}')(x)

    # Pointwise
    x = Conv2D(filters=int(pw_filters * alpha),
               kernel_size=1,
               strides=1,
               padding='same',
               use_bias=False,
               name=f'pointwise_conv_{block_id}')(x)
    if batch_norm:
        x = BatchNormalization(name=f'pointwise_bn_{block_id}')(x)
    x = ReLU(max_value=6.0, name=f'pointwise_relu6_{block_id}')(x)

    return x

def preprocess_input(inputs):
    '''
        Preprocess input for MobileNetV1
        Scale pixel values to [-1, 1]
    '''
    inputs /= 255.
    inputs = 2.0 * inputs - 1.0
    return inputs


class MobileNet():
    def __init__(self, input_shape=None, n_classes=None, alpha=1.0, batch_norm=True, dropout=None):
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
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.model = self.__build_model()

    def get_model(self):
        return self.model

    def __build_model(self):
        input = tf.keras.Input(shape=self.input_shape)

        depthwise_separable_stack = [
            {'strides': 1, 'pw_filters': 64, 'block_id': 1},
            {'strides': 2, 'pw_filters': 128, 'block_id': 2},
            {'strides': 1, 'pw_filters': 128, 'block_id': 3},
            {'strides': 2, 'pw_filters': 256, 'block_id': 4},
            {'strides': 1, 'pw_filters': 256, 'block_id': 5},
            {'strides': 2, 'pw_filters': 512, 'block_id': 6},
            [
                {'strides': 1, 'pw_filters': 512, 'block_id': i} for i in range(7, 7 + 5)
            ],
            {'strides': 2, 'pw_filters': 1024, 'block_id': 12},
            {'strides': 1, 'pw_filters': 1024, 'block_id': 13}
        ]

        # First conv layer
        x = _conv2d_block(input, filters=32, kernel_size=3, strides=2, alpha=self.alpha, batch_norm=self.batch_norm, block_id=1)

        # Depthwise separable blocks
        for block in depthwise_separable_stack:
            if isinstance(block, dict):
                x = _depthwise_separable_block(x, alpha=self.alpha, batch_norm=self.batch_norm, **block)
            else:
                for b in block:
                    x = _depthwise_separable_block(x, alpha=self.alpha, batch_norm=self.batch_norm, **b)

        # Average pooling & dropout
        x = GlobalAveragePooling2D(keepdims=True, name="avg_pool")(x)
        if self.dropout is not None:
            x = tf.keras.layers.Dropout(self.dropout, name="dropout")(x)

        # Output layer
        x = Dense(units=self.n_classes, activation='softmax', name="output")(x)

        return tf.keras.Model(inputs=input, outputs=x)
