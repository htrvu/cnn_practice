import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, AveragePooling2D, Dense, ReLU, MaxPooling2D, Dropout, Concatenate, Flatten

def _conv_block(inputs, n_filters, kernel_size, strides, padding, prefix=''):
    x = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding=padding, name=prefix)(inputs)
    x = BatchNormalization(name=f'{prefix}_bn')(x)
    x = ReLU(name=f'{prefix}_relu')(x)
    return x

def _inception_block(inputs, n_filters, prefix=''):
    # Conv 1x1
    # x1 = Conv2D(filters=n_filters[0], kernel_size=1, strides=1, padding='same', name=f'{prefix}_conv_1x1')(inputs)
    # x1 = BatchNormalization(name=f'{prefix}_conv_1x1_bn')(x1)
    # x1 = ReLU(name=f'{prefix}_conv_1x1_relu')(x1)
    x1 = _conv_block(inputs, n_filters[0], 1, 1, 'same', prefix=f'{prefix}_conv_1x1')

    # Conv 3x3
    x2 = _conv_block(inputs, n_filters[1][0], 1, 1, 'same', prefix=f'{prefix}_conv_3x3_reduce')
    x2 = _conv_block(x2, n_filters[1][1], 3, 1, 'same', prefix=f'{prefix}_conv_3x3')

    # Conv 5x5
    x3 = _conv_block(inputs, n_filters[2][0], 1, 1, 'same', prefix=f'{prefix}_conv_5x5_reduce')
    x3 = _conv_block(x3, n_filters[2][1], 5, 1, 'same', prefix=f'{prefix}_conv_5x5')

    # Pooling
    x4 = MaxPooling2D(pool_size=3, strides=1, padding='same', name=f'{prefix}_pool')(inputs)
    x4 = _conv_block(x4, n_filters[3], 1, 1, 'same', prefix=f'{prefix}_pool_proj')

    # Concatenate
    x = Concatenate(name=f'{prefix}_concat')([x1, x2, x3, x4])

    return x


def preprocess_input(inputs):
    '''
        Preprocess input for GoogLeNet (InceptionV1):
        - Subtract mean
    '''
    inputs = inputs - tf.constant([123.68, 116.779, 103.939], shape=[1, 1, 1, 3])
    return inputs


class GoogLeNet():
    def __init__(self, input_shape=None, n_classes=None, dropout=0.4):
        '''
            Args:
                input_shape: input shape of the image
                n_classes: number of classes
                dropout: dropout rate
        '''
        self.__input_shape = input_shape
        self.__n_classes = n_classes
        self.__dropout = dropout

        self.__model = self.__build_model()

    def get_model(self):
        return self.__model

    def __build_model(self):
        input = tf.keras.Input(shape=self.__input_shape)
        x = input

        # First conv
        x = _conv_block(x, 64, 7, 2, 'same', prefix='conv_1')
        x = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool_1')(x)
        x = tf.nn.local_response_normalization(x, name='lrn_1')

        # Second conv
        x = _conv_block(x, 192, 1, 1, 'same', prefix='conv_2')
        x = tf.nn.local_response_normalization(x, name='lrn_2')
        x = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool_2')(x)

        # Inception blocks
        x = _inception_block(x, n_filters=[64, (96, 128), (16, 32), 32], prefix='inception_3a')
        x = _inception_block(x, n_filters=[128, (128, 192), (32, 96), 64], prefix='inception_3b')
        x = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool_3')(x)
        x = _inception_block(x, n_filters=[192, (96, 208), (16, 48), 64], prefix='inception_4a')
        x = _inception_block(x, n_filters=[160, (112, 224), (24, 64), 64], prefix='inception_4b')
        x = _inception_block(x, n_filters=[128, (128, 256), (24, 64), 64], prefix='inception_4c')
        x = _inception_block(x, n_filters=[112, (144, 288), (32, 64), 64], prefix='inception_4d')
        x = _inception_block(x, n_filters=[256, (160, 320), (32, 128), 128], prefix='inception_4e')
        x = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool_4')(x)
        x = _inception_block(x, n_filters=[256, (160, 320), (32, 128), 128], prefix='inception_5a')
        x = _inception_block(x, n_filters=[384, (192, 384), (48, 128), 128], prefix='inception_5b')

        # Average pooling
        x = AveragePooling2D(pool_size=7, strides=1, name='pool_5')(x)

        # Dropout
        if self.__dropout is not None:
            x = Dropout(self.__dropout, name='dropout')(x)

        # Output
        x = Flatten(name='flatten')(x)
        x = Dense(self.__n_classes, activation='softmax', name='output')(x)
                
        return tf.keras.Model(inputs=input, outputs=x)


