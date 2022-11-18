import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense, ReLU, AveragePooling2D, Dropout, Concatenate


def _bottleneck_layers(inputs, growth_rate, prefix='', block_id=''):
    '''
        Bottleneck layers for DenseNet        
    '''
    x = BatchNormalization(name=f'{prefix}_bn_{block_id}_1')(inputs)
    x = ReLU(name=f'{prefix}_relu_{block_id}_1')(x)
    x = Conv2D(filters=4*growth_rate, kernel_size=1, strides=1, padding='same', name=f'{prefix}_conv_{block_id}_1', use_bias=False)(x)
    x = BatchNormalization(name=f'{prefix}_bn_{block_id}_2')(x)
    x = ReLU(name=f'{prefix}_relu_{block_id}_2')(x)
    x = Conv2D(filters=growth_rate, kernel_size=3, strides=1, padding='same', name=f'{prefix}_conv_{block_id}_2', use_bias=False)(x)
    return x


def _dense_block(inputs, n_bb_layers, growth_rate, prefix='', block_id=''):
    '''
        Dense block for DenseNet
    '''
    x = inputs
    for i in range(n_bb_layers):
        bb_output = _bottleneck_layers(x, growth_rate, prefix=prefix, block_id=f'{block_id}_{i}')
        x = Concatenate(axis=-1, name=f'{prefix}_concat_{i+1}')([bb_output, x])
    return x


def _transition_layers(inputs, theta, prefix='', block_id=''):
    '''
        Transition layers for DenseNet
    '''
    n_filters = int(theta * inputs.shape[-1])
    x = BatchNormalization(name=f'{prefix}_bn_{block_id}_1')(inputs)
    x = ReLU(name=f'{prefix}_relu_{block_id}_1')(x)
    x = Conv2D(filters=n_filters, kernel_size=1, strides=1, padding='same', name=f'{prefix}_conv_{block_id}_1', use_bias=False)(x)
    x = AveragePooling2D(pool_size=2, strides=2, padding='same', name=f'{prefix}_pool_{block_id}_1')(x)
    return x


def preprocess_input(inputs):
    '''
        Preprocess input for DenseNet:
        - Scale the input to [0, 1]
        - Normalize each channel with respect to the ImageNet dataset
    '''
    inputs = inputs / 255.0
    mean = tf.constant([0.485, 0.456, 0.406], dtype=inputs.dtype, shape=[1, 1, 1, 3])
    std = tf.constant([0.229, 0.224, 0.225], dtype=inputs.dtype, shape=[1, 1, 1, 3])
    inputs = (inputs - mean) / std
    return inputs


CONFIGS = {
    'densenet121': {
        'n_bb_layers': [6, 12, 24, 16],
        'growth_rate': 32,
        'theta': 0.5
    },
    'densenet169': {
        'n_bb_layers': [6, 12, 32, 32],
        'growth_rate': 32,
        'theta': 0.5
    },
    'densenet201': {
        'n_bb_layers': [6, 12, 48, 32],
        'growth_rate': 32,
        'theta': 0.5
    },
    'densenet264': {
        'n_bb_layers': [6, 12, 64, 48],
        'growth_rate': 32,
        'theta': 0.5
    }
}


class DenseNet():
    def __init__(self, model_name, input_shape=None, n_classes=None, dropout=None):
        '''
            Args:
                model_name: str, name of the model (vgg11, vgg13, vgg16, vgg19)
                input_shape: input shape of the image
                n_classes: number of classes
                theta: width multiplier
                batch_norm: whether to use batch normalization
                dropout: dropout rate
        '''
        if model_name not in CONFIGS:
                raise ValueError(f'Invalid model name: {model_name}')

        self.__input_shape = input_shape
        self.__n_classes = n_classes
        self.__dropout = dropout
        self.__config = CONFIGS[model_name]

        self.__model = self.__build_model()

    def get_model(self):
        return self.__model

    def __build_model(self):
        input = tf.keras.Input(shape=self.__input_shape)
        x = input

        n_bb_layers, growth_rate, theta = self.__config['n_bb_layers'], self.__config['growth_rate'], self.__config['theta']

        x = Conv2D(filters=2*growth_rate, kernel_size=7, strides=2, padding='same', name='conv_1', use_bias=False)(x)
        x = BatchNormalization(name='bn_1')(x)
        x = ReLU(name='relu_1')(x)
        x = AveragePooling2D(pool_size=3, strides=2, padding='same')(x)

        for i, n_bb_layer in enumerate(n_bb_layers):
            x = _dense_block(x, n_bb_layer, growth_rate, prefix=f'dense_block_{i + 1}', block_id=i)
            if i != len(n_bb_layers) - 1:
                x = _transition_layers(x, theta, prefix=f'transition_layer_{i + 1}', block_id=i)

        x = BatchNormalization(name='bn_last')(x)
        x = ReLU(name='relu_last')(x)

        x = GlobalAveragePooling2D(name='avg_pool')(x)
        if self.__dropout:
            x = Dropout(self.__dropout)(x)

        x = Dense(self.__n_classes, activation='softmax', name='fc')(x)

        return tf.keras.Model(inputs=input, outputs=x)


def DenseNet121(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return DenseNet('densenet121', input_shape=input_shape, n_classes=n_classes, dropout=dropout)

def DenseNet169(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return DenseNet('densenet169', input_shape=input_shape, n_classes=n_classes, dropout=dropout)

def DenseNet201(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return DenseNet('densenet201', input_shape=input_shape, n_classes=n_classes, dropout=dropout)

def DenseNet264(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return DenseNet('densenet264', input_shape=input_shape, n_classes=n_classes, dropout=dropout)