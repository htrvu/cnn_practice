import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense, ReLU, MaxPooling2D, Dropout

def _conv2d_block(input, filters, kernel_size, strides, use_af=True, prefix='', block_id=''):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               name=f'{prefix}_conv_{block_id}')(input)
    x = BatchNormalization(name=f'{prefix}_bn_{block_id}')(x)
    if use_af:
        x = ReLU(max_value=6.0, name=f'{prefix}_relu6_{block_id}')(x)
    return x

def preprocess_input(inputs):
    '''
        Preprocess input for ResNet:

        - Convert the images from RGB to BGR
        - Zero-center each color channel with respect to the ImageNet dataset, without scaling.
    '''
    inputs = inputs[..., ::-1]
    inputs[..., 0] -= 103.939
    inputs[..., 1] -= 116.779
    inputs[..., 2] -= 123.68
    return inputs


def ResBlock(input, filters, strides, block_id):
    prefix = f'res_block_{block_id}'
    x = _conv2d_block(input, filters, 3, strides, prefix=prefix, block_id=1)
    x = _conv2d_block(x, filters, 3, 1, prefix=prefix, block_id=2)

    if input.shape[-1] != filters:
        # projection shortcut
        prefix = f'res_block_{block_id}_projection'
        input = _conv2d_block(input, filters, 1, strides, use_af=False, prefix=prefix, block_id=0)

    x = tf.keras.layers.Add(name=f'{prefix}_add')([input, x])
    x = ReLU(max_value=6.0, name=f'{prefix}_relu6_add')(x)
    return x

def ResStack(input, filters, n_blocks, first_strides, stack_id):
    x = ResBlock(input, filters=filters, strides=first_strides, block_id=f'{stack_id}_{1}')
    for i in range(2, n_blocks + 1):
        x = ResBlock(x, filters=filters, strides=1, block_id=f'{stack_id}_{i}')
    return x


def BottleneckBlock(input, filters, strides, block_id):
    prefix = f'bottleneck_block_{block_id}'
    x = _conv2d_block(input, filters, 1, strides, prefix=prefix, block_id=1)
    x = _conv2d_block(x, filters, 3, 1, prefix=prefix, block_id=2)
    x = _conv2d_block(x, 4 * filters, 1, 1, prefix=prefix, block_id=3)
    
    if input.shape[-1] != 4 * filters:
        # projection shortcut
        prefix = f'bottleneck_block_{block_id}_projection'
        input = _conv2d_block(input, 4 * filters, 1, strides, use_af=False, prefix=prefix, block_id='')
    
    x = tf.keras.layers.add([input, x])
    x = ReLU(max_value=6.0, name=f'{prefix}_relu6_add')(x)
    return x

def BottleneckStack(input, filters, n_blocks, first_strides, stack_id):
    x = BottleneckBlock(input, filters=filters, strides=first_strides, block_id=f'{stack_id}_{1}')
    for i in range(2, n_blocks + 1):
        x = BottleneckBlock(x, filters=filters, strides=1, block_id=f'{stack_id}_{i}')
    return x


CONFIGS = {
    'resnet18': {
        'n_blocks': [2, 2, 2, 2],
        'filters': [64, 128, 256, 512],
        'first_strides': [1, 2, 2, 2],
        'create_func': ResStack
    },
    'resnet34': {
        'n_blocks': [3, 4, 6, 3],
        'filters': [64, 128, 256, 512],
        'first_strides': [1, 2, 2, 2],
        'create_func': ResStack
    },
    'resnet50': {
        'n_blocks': [3, 4, 6, 3],
        'filters': [64, 128, 256, 512],
        'first_strides': [1, 2, 2, 2],
        'create_func': BottleneckStack
    },
    'resnet101': {
        'n_blocks': [3, 4, 23, 3],
        'filters': [64, 128, 256, 512],
        'first_strides': [1, 2, 2, 2],
        'create_func': BottleneckStack
    },
    'resnet152': {
        'n_blocks': [3, 8, 36, 3],
        'filters': [64, 128, 256, 512],
        'first_strides': [1, 2, 2, 2],
        'create_func': BottleneckStack
    }
}


class ResNet():
    def __init__(self, model_name, input_shape=None, n_classes=None, dropout=None):
        '''
            Args:
                model_name: str, name of the model (resnet18, resnet34, resnet50, resnet101, resnet152)
                input_shape: input shape of the image
                n_classes: number of classes
                alpha: width multiplier
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
        x = _conv2d_block(input, 64, 7, 2, prefix='conv1', block_id=1)
        x = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool1')(x)

        n_blocks, filters, first_strides, create_fnc = self.__config['n_blocks'], self.__config['filters'], self.__config['first_strides'], self.__config['create_func']
        for i in range(len(n_blocks)):
            x = create_fnc(x, filters=filters[i], n_blocks=n_blocks[i], first_strides=first_strides[i], stack_id=i+1)
        
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        if self.__dropout is not None:
            x = Dropout(self.__dropout)(x)
        x = Dense(self.__n_classes, activation='softmax', name='output')(x)

        return tf.keras.Model(inputs=input, outputs=x)
        

def ResNet18(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return ResNet('resnet18', input_shape=input_shape, n_classes=n_classes, dropout=dropout)

def ResNet34(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return ResNet('resnet34', input_shape=input_shape, n_classes=n_classes, dropout=dropout)

def ResNet50(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return ResNet('resnet50', input_shape=input_shape, n_classes=n_classes, dropout=dropout)

def ResNet101(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return ResNet('resnet101', input_shape=input_shape, n_classes=n_classes, dropout=dropout)

def ResNet152(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return ResNet('resnet152', input_shape=input_shape, n_classes=n_classes, dropout=dropout)




