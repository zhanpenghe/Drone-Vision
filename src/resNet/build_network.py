'''
    Build the deep network for drone direction determination.
    This is based on https://arxiv.org/pdf/1705.02550.pd
'''

from keras.models import Model
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, AveragePooling2D, Activation, Dropout, BatchNormalization
from keras.layers.merge import add, concatenate
from keras import regularizers
#from keras_contrib.layers.advanced_activations import SReLU
from keras import backend

def convRelu(filters, kernel_size, strides, padding='same'):
    def func(prev_layer):
        convolution = BatchNormalization(axis=3)(Conv2D(filters=filters, kernel_size=kernel_size,
                             strides=strides, padding=padding)(prev_layer))
        return Activation('relu')(convolution)

    return func


# Convolution->SReLU()->Convolution, both convolution are 3*3 with stride of 1
def blockFunc(filters, kernel_size, init_strides=(2,2), padding='same'):
    def func(prev_layer):
        conv_1 = convRelu(filters=filters, kernel_size=kernel_size, strides=init_strides, padding=padding)(prev_layer)
        conv_1 = Dropout(0.1)(conv_1)
        return Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding=padding, kernel_regularizer=regularizers.l2(1e-4))(conv_1)				#todo check regularizer 

    return func


def shortcut(prev_layer, residual):
    residual_shape = backend.int_shape(residual)
    input_shape = backend.int_shape(prev_layer)

    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))

    if input_shape[3]!=residual_shape[3] or stride_width>1 or stride_height>1:
        prev_layer = Conv2D(filters=residual_shape[3], kernel_size=(1, 1), strides=(stride_width, stride_height), padding="valid")(prev_layer)

    return add([prev_layer, residual])


# helper to build a residual block
def resBlock(filters, strides, isFirstBlock=False):
    def single_block(prev_layer, isFirstLayer=False):
        #bn = BatchNormalization(axis=3)(prev_layer)
        if isFirstLayer:
            conv_1 = blockFunc(filters=filters, kernel_size=(3, 3), init_strides=(2,2))(prev_layer)
        else:
            conv_1 = blockFunc(filters=filters, kernel_size=(3, 3), init_strides=(1,1))(prev_layer)
        res = shortcut(prev_layer, conv_1)
        res = BatchNormalization(axis=3)(res)
        return Activation('relu')(res)

    def func(prev_layer):
        if isFirstBlock:
            return single_block(single_block(prev_layer, isFirstLayer=True))

        block = single_block(single_block(prev_layer, isFirstLayer=True))
        return block

    return func


def buildNetwork(input_shape, output_shape):
    # Check input shape dimension
    if len(input_shape) != 3:
        raise Exception('Input shape is not correct! (nb_rows, nb_cols, nb_channels)')

    # input->conv->relu->max_pooling
    input_layer = Input(shape=input_shape)
    conv1_pooling = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(
        (convRelu(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input_layer)))

    block = conv1_pooling
    # resNet
    filters = 64
    for i in range(4):
        block = resBlock(filters=filters, strides=(1, 1), isFirstBlock=(i == 0))(block)
        filters *= 2

    block_shape = backend.int_shape(block)
    average_pooling = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]), strides=None)(block)

    flatten1 = Flatten()(average_pooling)
    dense = BatchNormalization()(flatten1)

    dense = Dense(512)(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    
    dense = Dropout(0.5)(dense)

    dense = Dense(256)(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)

    dense = Dropout(0.5)(dense)

    dense1 = Dense(64)(dense)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(3)(dense1)
    dense1_ = Activation('softmax')(dense1)


    '''dense2 = Dense(64)(dense)
                dense2 = BatchNormalization()(dense2)
                dense2 = Activation('relu')(dense2)
                dense2 = Dropout(0.5)(dense2)
                dense2 = Dense(3)(dense2)
                dense2_ = Activation('softmax')(dense2)'''
    

    return Model(inputs=input_layer, outputs=dense1_)

def build_network_for_hires(input_shape, output_shape):
    # Check input shape dimension
    if len(input_shape) != 3:
        raise Exception('Input shape is not correct! (nb_rows, nb_cols, nb_channels)')

    # input->conv->relu->max_pooling
    input_layer = Input(shape=input_shape)
    conv1_pooling = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(
        (convRelu(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input_layer)))

    block = conv1_pooling
    # resNet
    filters = 64
    for i in range(4):
        block = resBlock(filters=filters, strides=(1, 1), isFirstBlock=(i == 0))(block)
        filters *= 2

    block_shape = backend.int_shape(block)
    average_pooling = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]), strides=None)(block)

    flatten1 = Flatten()(average_pooling)
    dense = BatchNormalization()(flatten1)

    dense = Dense(512)(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    
    dense = Dropout(0.5)(dense)

    dense = Dense(256)(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)

    dense = Dropout(0.5)(dense)

    dense1 = Dense(64)(dense)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense1 = Dense(3)(dense1)
    dense1_ = Activation('softmax')(dense1)


    return Model(inputs=input_layer, outputs=dense1_)

def saveVisualizedModel(model, filepath):
    from keras.utils import plot_model
    plot_model(model, to_file=filepath, show_shapes=True)

def test():
    model, _ = buildNetwork((256, 256, 3), 11)
    print(model.summary())
    saveVisualizedModel(model, 'model.png')
    yaml = model.to_yaml()
    #print(yaml)

#test()
