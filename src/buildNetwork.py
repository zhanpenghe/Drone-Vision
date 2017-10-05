'''
    Build the deep network for drone direction determination.
    This is based on https://arxiv.org/pdf/1705.02550.pd

    todo #1 downsampling at residual block
    todo #2 change relu to shifted relu
    todo #3 output layers
    todo #4 data pre-processing
'''

from keras.models import Model
from keras.layers import Dense, Conv2D, Input, Activation, MaxPool2D, Flatten
from keras.layers.merge import add


def convRelu(filters, kernel_size, strides, padding='same'):
    def func(input):
        convolution = Conv2D(filters=filters, kernel_size=kernel_size,
                             strides=strides, padding=padding)(input)
        return Activation('relu')(convolution)

    return func


# Convolution->SReLU->Convolution, both convolution are 3*3 with stride of 1
def blockFunc(filters, kernel_size, strides, padding='same'):
    def func(input):
        conv_1 = convRelu(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input)
        return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(conv_1)

    return func


def shortcut(input, residual):
    return add([input, residual])


# helper to build a residual block
def resBlock(filters, strides, isFirstBlock=False):
    def single_block(input):
        conv_1 = blockFunc(filters=filters, kernel_size=(3, 3), strides=1)(input)
        res = shortcut(input, conv_1)
        return Activation('relu')(res)

    def func(input):
        if isFirstBlock:
            return single_block(single_block(input))

        block = blockFunc(filters=filters, kernel_size=(3, 3), strides=strides)(input)
        block = Activation('relu')(block)
        return single_block(block)

    return func


def buildNetwork(input_shape):
    # Check input shape dimension
    if len(input_shape) != 3:
        raise Exception('Input shape is not correct! (nb_rows, nb_cols, nb_channels)')

    # input->conv->relu->max_pooling
    input = Input(shape=input_shape)
    conv1_pooling = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(
        (convRelu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)))

    block = conv1_pooling
    # resNet
    filters = 64
    for i in range(4):
        block = resBlock(filters=filters, strides=1, isFirstBlock=(i == 0))(block)
        filters *= 2

    flatten1 = Flatten()(block)
    dense = Dense(units=512, kernel_initializer="he_normal",
                  activation="softmax")(flatten1)

    return Model(inputs=input, outputs=dense)


def test():
    model = buildNetwork((320, 180, 3))
    for layer in model.layers:
        print(type(layer))
        print(layer.output_shape)


test()
