#
# Created on Thu Jun 03 2021 10:25:56 AM
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
# Objective : Test a trained model
# ////// libraries ///// 
# standard 
import numpy as np
from keras.optimizers import Adam

# internal 
from src.data.utils import load_phase_data
from src.visualization.utils import plot, plot_hist
from src.models.architectures import JointConvSQDLSTMNet

# ////// body /////

## user inputs
use_JointConvSQDLSTMNet = True
model_id = "Model_JointConvSQDLSTMNet_Noisy_Phase_Data_1000_8pi_8pi"
dataset_id = "Noisy_Phase_Data_400_8pi_8pi" # testing dataset
image_size = (256, 256, 1) 
batch_size = 4

## load test data and visualize a pair
X_test, y_test = load_phase_data(dataset_id)
print("Number of Testing Samples : {:n}".format(X_test.shape[0]))
idx = np.random.randint(0, X_test.shape[0])
plot(X_test[idx], y_test[idx], titles=["Noisy Wrapped Phase ($\psi$)", "True Phase ($\phi$)"])

## load trained model
if use_JointConvSQDLSTMNet:
    model = JointConvSQDLSTMNet(image_size).getModel()
    model.summary()

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    metrics=[]
)

model_path = 'DeepPhaseUnwrap/models/{}.h5'.format(model_id)
model.load_weights(model_path)

## predict true phase
y_pred = model.predict(X_test, batch_size=batch_size)

## get the scaled predicted true phase values
y_pred_scaled = np.empty((0, 256, 256))
for i in range(X_test.shape[0]):
  Xi = X_test[i]
  yi = y_test[i]
  ypi = y_pred[i]
  
  # match scales of predicted true phase
  min1, max1 = np.min(yi), np.max(yi)
  min2, max2 = np.min(ypi), np.max(ypi)
  temp = (ypi - min2) / (max2 - min2)
  ypi_scaled = temp * (max1 - min1) + min1
  y_pred_scaled = np.vstack((y_pred_scaled, ypi_scaled.reshape(1, 256, 256)))

## compute Normalize Root Mean Squared Error
error = y_test - y_pred_scaled
r = np.max(y_test, axis=(1, 2), keepdims=True) - np.min(y_test, axis=(1, 2), keepdims=True)
NRMSE = np.mean(np.sqrt(np.mean(error**2, axis=(1, 2)))/r)*100
performance = "NRMSE = {:.2f} %".format(NRMSE)
print(performance)

## visualize some predicted phase maps
indices = np.random.randint(0, X_test.shape[0], size=(10, ))
for i in range(10):
  Xi = X_test[indices[i]]
  yi = y_test[indices[i]]
  ypi = y_pred_scaled[indices[i]]

  # visualize
  plot(Xi, yi, ypi_scaled, titles=["Noisy Wrapped Phase ($\psi$)", "True Phase ($\phi$)", "Predicted True Phase ($\hat{\phi}$)"])
  plot_hist(yi, ypi_scaled, titles=["True Phase ($\phi$)", "Predicted True Phase ($\hat{\phi}$)"])
