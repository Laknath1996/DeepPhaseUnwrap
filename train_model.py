#
# Created on Thu Jun 03 2021 10:00:06 AM
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
# Objective : Train a model

# ////// libraries ///// 
# standard 
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

# internal
from src.data.utils import load_phase_data 
from src.visualization.utils import plot
from src.models.architectures import JointConvSQDLSTMNet
from src.models.losses import tv_loss_plus_var_loss

# ////// body ///// 

## user inputs
use_JointConvSQDLSTMNet = True
dataset_id = "Noisy_Phase_Data_1000_8pi_8pi"
image_size = (256, 256, 1)
lr = 1e-3
loss = tv_loss_plus_var_loss
batch_size = 4
epochs = 100

## load phase data and visualize a pair
X, y = load_phase_data(dataset_id)
print("Number of Training Samples : {:n}".format(X.shape[0]))
idx = np.random.randint(0, X.shape[0])
plot(X[idx], y[idx], titles=["Noisy Wrapped Phase ($\psi$)", "True Phase ($\phi$)"])

## get model
if use_JointConvSQDLSTMNet:
    network_id = 'JointConvSQDLSTMNet'
    model = JointConvSQDLSTMNet(image_size).getModel()
    model.summary()

## define model specs
model_id = 'Model_{}_{}'.format(network_id,dataset_id)
model_path = 'DeepPhaseUnwrap/models/{}.h5'.format(model_id)

model.compile(
    optimizer=Adam(learning_rate=lr),
    loss=loss
)

earlystopper = EarlyStopping(
    monitor='loss',
    patience=10,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    model_path,
    monitor='loss',
    verbose=1,
    save_best_only=True
)

## train the model
history = model.fit(
    x = X.reshape(X.shape[0], image_size[0], image_size[1], image_size[2]),
    y = y.reshape(y.shape[0], image_size[0], image_size[1], image_size[2]),
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    callbacks=[model_checkpoint, earlystopper]   
)

## plot epochs vs loss
loss = history.history['loss']
epochs = np.arange(0, len(loss), 1)
plt.plot(epochs, loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epochs vs. Loss")
plt.grid()
plt.show