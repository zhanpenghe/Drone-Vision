'''
    Build the deep network for drone direction determination.
    This is based on https://arxiv.org/pdf/1705.02550.pd

    todo data pre-processing
'''

from keras.models import Model
from keras.layers import Dense, Conv2D, Input, MaxPool2D, Flatten, AveragePooling2D
from keras.layers.merge import add
from keras_contrib.layers.advanced_activations import SReLU
from keras import backend

def convRelu(filters, kernel_size, strides, padding='same'):
    def func(input):
        convolution = Conv2D(filters=filters, kernel_size=kernel_size,
                             strides=strides, padding=padding)(input)
        return SReLU()(convolution)

    return func


# Convolution->SReLU()->Convolution, both convolution are 3*3 with stride of 1
def blockFunc(filters, kernel_size, strides, padding='same'):
    def func(input):
        conv_1 = convRelu(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(input)
        return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(conv_1)

    return func


def shortcut(input, residual):
    residual_shape = backend.int_shape(residual)
    input_shape = backend.int_shape(input)

    if input_shape[3]!=residual_shape[3]:
        stride_width = int(round(input_shape[1] / residual_shape[1]))
        stride_height = int(round(input_shape[2] / residual_shape[2]))
        input = Conv2D(filters=residual_shape[3], kernel_size=(1, 1), strides=(stride_width, stride_height), padding="valid")(input)

    return add([input, residual])


# helper to build a residual block
def resBlock(filters, strides, isFirstBlock=False):
    def single_block(input):
        conv_1 = blockFunc(filters=filters, kernel_size=(3, 3), strides=1)(input)
        res = shortcut(input, conv_1)
        return SReLU()(res)

    def func(input):
        if isFirstBlock:
            return single_block(single_block(input))

        #downsampling first
        block = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(input)
        block = single_block(single_block(block))
        return block

    return func


def buildNetwork(input_shape, output_shape):
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
        block = resBlock(filters=filters, strides=(1, 1), isFirstBlock=(i == 0))(block)
        filters *= 2

    block_shape = backend.int_shape(block)
    average_pooling = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]), strides=None)(block)

    flatten1 = Flatten()(average_pooling)
    dense = Dense(output_shape)(flatten1)
    return Model(inputs=input, outputs=dense)

def saveVisualizedModel(model, filepath):
    from keras.utils import plot_model
    plot_model(model, to_file=filepath, show_shapes=True)

def test():
    model = buildNetwork((320, 180, 3))
    print(model.summary())
    #saveVisualizedModel(model)
    yaml = model.to_yaml()
    #print(yaml)

#test()

