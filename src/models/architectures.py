#
# Created on Thu Jun 03 2021 9:00:38 AM
#
# The MIT License (MIT)
# Copyright (c) 2021 Ashwin De Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Objective : Define the deep neural architectures

# ////// libraries ///// 
# standard 
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, concatenate
from keras.layers import Input, AveragePooling2D, Reshape, Permute
from keras.layers import Bidirectional, LSTM
from keras.models import Model
import keras.backend as K
K.set_image_data_format('channels_last')

# ////// body ///// 

class JointConvSQDLSTMNet():
    """
    The joint convolutional and spatial quad-directional LSTM network proposed by
    M. V. Perera and A. De Silva, "A Joint Convolutional and Spatial Quad-Directional 
    LSTM Network for Phase Unwrapping," ICASSP 2021 - 2021 IEEE International Conference 
    on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 4055-4059, 
    doi: 10.1109/ICASSP39728.2021.9414748.
    """
    def __init__(self, input_shape):
        super(JointConvSQDLSTMNet, self).__init__()
        self.input_shape = input_shape

    def getModel(self):
        """
        Defines the joint convoltional and spatial quad-directional LSTM network
        """
        ## input to the network
        input = Input(self.input_shape)

        ## encoder network
        c1 = Conv2D(filters=16, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(input)
        c1 = BatchNormalization()(c1)
        c1 = Activation('relu')(c1)
        p1 = AveragePooling2D()(c1)

        c2 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Activation('relu')(c2)
        p2 = AveragePooling2D()(c2)

        c3 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Activation('relu')(c3)
        p3 = AveragePooling2D()(c3)

        c4 = Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Activation('relu')(c4)
        p4 = AveragePooling2D()(c4)

        # SQD-LSTM Block
        x_hor_1 = Reshape((16 * 16, 128))(p4)
        x_ver_1 = Reshape((16 * 16, 128))(Permute((2, 1, 3))(p4))

        h_hor_1 = Bidirectional(LSTM(units=32, activation='tanh', return_sequences=True, go_backwards=False))(x_hor_1)
        h_ver_1 = Bidirectional(LSTM(units=32, activation='tanh', return_sequences=True, go_backwards=False))(x_ver_1)

        H_hor_1 = Reshape((16, 16, 64))(h_hor_1)
        H_ver_1 = Permute((2, 1, 3))(Reshape((16, 16, 64))(h_ver_1))

        c_hor_1 = Conv2D(filters=64, kernel_size=(3, 3),
                        kernel_initializer='he_normal', padding='same')(H_hor_1)
        c_ver_1 = Conv2D(filters=64, kernel_size=(3, 3),
                        kernel_initializer='he_normal', padding='same')(H_ver_1)

        H = concatenate([c_hor_1, c_ver_1])

        # decoder Network
        u5 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(H)
        u5 = concatenate([u5, c4])
        c5 = Conv2D(filters=128, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u5)
        c5 = BatchNormalization()(c5)
        c5 = Activation('relu')(c5)

        u6 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c3])
        c6 = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Activation('relu')(c6)

        u7 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c2])
        c7 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Activation('relu')(c7)

        u8 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c1])
        c8 = Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Activation('relu')(c8)

        ## output layer
        output = Conv2D(filters=1, kernel_size=(1, 1), padding='same', name='out1')(c8)
        output = Activation('linear')(output)

        ## define the model
        model = Model(inputs=[input], outputs=[output])
        return model
