#
# Created on Thu Jun 03 2021 9:30:52 AM
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
# Objective : Define loss functions

# ////// libraries ///// 
# standard 
import keras.backend as K
K.set_image_data_format('channels_last')

# ////// body ///// 

def tv_loss_plus_var_loss(y_true, y_pred):
    """
    Define the composite loss function that includes total variation of errors 
    loss and variance of errors loss
    """
    # total variation loss
    y_x = y_true[:, 1:256, :, :] - y_true[:, 0:255, :, :]
    y_y = y_true[:, :, 1:256, :] - y_true[:, :, 0:255, :]
    y_bar_x = y_pred[:, 1:256, :, :] - y_pred[:, 0:255, :, :]
    y_bar_y = y_pred[:, :, 1:256, :] - y_pred[:, :, 0:255, :]
    L_tv = K.mean(K.abs(y_x - y_bar_x)) + K.mean(K.abs(y_y - y_bar_y))

    # variance of the error loss
    E = y_pred - y_true
    L_var = K.mean(K.mean(K.square(E), axis=(1, 2, 3)) - K.square(K.mean(E, axis=(1, 2, 3))))

    loss = L_var + 0.1 * L_tv
    return loss
