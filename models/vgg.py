import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Dense, ReLU, MaxPooling2D, Flatten

def VGGBlock(input, n_filters):
    x = input
    for filters in n_filters:
        x = Conv2D(filters,kernel_size=3, strides=1, padding='same')(x)
        x = ReLU()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    return x

CONFIGS = {
    'vgg11': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'vgg13': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'vgg16': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'vgg19': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
}

def preprocess_input(inputs):
    '''
        Preprocess input for VGG:

        - Convert the images from RGB to BGR
        - Zero-center each color channel with respect to the ImageNet dataset, without scaling.
    '''
    inputs = inputs[..., ::-1]
    mean = tf.constant([103.939, 116.779, 123.68], dtype=inputs.dtype, shape=[1, 1, 1, 3])
    inputs = inputs - mean
    return inputs


class VGG():
    def __init__(self, model_name, input_shape=None, n_classes=None, dropout=None):
        '''
            Args:
                model_name: str, name of the model (vgg11, vgg13, vgg16, vgg19)
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
        x = input

        for n_filters in self.__config:
            x = VGGBlock(x, n_filters)

        x = Flatten()(x)
        if self.__dropout is not None:
            x = Dropout(self.__dropout)(x)

        x = Dense(4096, activation='relu')(x)
        x = Dense(4096, activation='relu')(x)
        x = Dense(self.__n_classes, activation='softmax')(x)

        return tf.keras.Model(inputs=input, outputs=x)

        
def VGG11(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return VGG('vgg11', input_shape=input_shape, n_classes=n_classes, dropout=dropout)

def VGG13(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return VGG('vgg13', input_shape=input_shape, n_classes=n_classes, dropout=dropout)

def VGG16(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return VGG('vgg16', input_shape=input_shape, n_classes=n_classes, dropout=dropout)

def VGG19(input_shape=(224, 224, 3), n_classes=1000, dropout=None):
    return VGG('vgg19', input_shape=input_shape, n_classes=n_classes, dropout=dropout)