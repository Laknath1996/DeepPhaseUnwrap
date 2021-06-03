#
# Created on Thu Jun 03 2021 9:40:53 AM
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
# Objective : Create Synthetic Datasets

# ////// libraries ///// 
# standard 
import numpy as np
import h5py
import os
from scipy.stats import norm

# ////// body ///// 
def simulate(size, m_1, m_2, C, A, mu_x, mu_y, sigma_x, sigma_y):
  """
  creates an arbitrary phase map by mixing gaussian blobs and adding ramps
  """
  x = np.arange(0, size[0], 1)
  y = np.arange(0, size[0], 1)
  xx, yy = np.meshgrid(x, y, sparse=True)
  I = np.zeros(size)
  ## mix randomly shaped and placed gaussian blobs
  for i in range(len(sigma_x)):
      a = (xx-mu_x[i])**2/(2*sigma_x[i]**2) + (yy-mu_y[i])**2/(2*sigma_y[i]**2)
      I += A[i]*np.exp(-a)
  ## add ramp phase with random gradients and shifts
  I = m_1*xx + m_2*yy + C + 0.1*I
  return I

def wrap(phi):
  """
  wraps the true phase signal within [-pi, pi]
  """
  return np.angle(np.exp(1j*phi))

def rescale(im, range):
  """
  mini-max rescales the input image
  """
  im_std = (im - im.min()) / (im.max() - im.min())
  im_scaled = im_std * (range[1] - range[0]) + range[0]
  return im_scaled

def create_random_image(size):
  """
  creates an randomly simulated true phase map
  """ 
  array_len = np.random.randint(2, 5)
  m = np.random.uniform(0, 0.5, [2])
  C = np.random.randint(1, 10)
  A = np.random.randint(50, 1000, array_len)
  mu_x = np.random.randint(20, 235, array_len)
  mu_y = np.random.randint(20, 235, array_len)
  sigma_x = np.random.randint(10, 45, array_len)
  sigma_y = np.random.randint(10, 45, array_len)
  I = simulate(size, m[0], m[1], C, A, mu_x, mu_y, sigma_x, sigma_y)
  return I

def create_dataset(path, size, no_samples, max_lower_bound, max_upper_bound, noise_levels):
  """
  creates the synthetic true-wrapped phase dataset
  """
  wrapped_phase_maps = np.zeros((1, size[0], size[1]))
  true_phase_maps = np.zeros((1, size[0], size[1])) 

  ## create dataset
  for i in range(no_samples):
      print("Creating {:n}/{:n} pairs".format(i+1, no_samples))

      ## generate the true and wrapped phase maps
      I = create_random_image(size)
      lower_bound = (-2) * np.pi * np.random.randint(1, max_lower_bound+1) 
      upper_bound = 2 * np.pi * np.random.randint(1, max_upper_bound+1) 
      I = rescale(I, [lower_bound, upper_bound])
      I_wrap = wrap(I)

      ## adding noise to the true phase before wrapping it
      snr = noise_levels[np.random.randint(0, len(noise_levels))]
      reqSNR = 10**(snr/10)
      sigPower = 1
      sigPower = 10**(sigPower/10)
      noisePower = sigPower/reqSNR
      I_gaun = np.sqrt(noisePower)*norm.rvs(0, 1, size=(256, 256)) # gaussian noise
      I_n = I + I_gaun # noisy true phase
      I_wrap_n = wrap(I_n) # noisy wrapped phase

      wrapped_phase_maps = np.concatenate((wrapped_phase_maps, I_wrap_n.reshape(1, size[0], size[1])), axis=0)
      true_phase_maps = np.concatenate((true_phase_maps, I.reshape(1, size[0], size[1])), axis=0)

  ## save dataset
  dataset_id = "Noisy_Phase_Data_{:n}_{:n}pi_{:n}pi.hdf5".format(no_samples, 2*max_lower_bound, 2*max_upper_bound)
  dataset = h5py.File(os.path.join(path, dataset_id), mode='w')
  dataset.create_dataset('psi', data=wrapped_phase_maps[1:, ...])
  dataset.create_dataset('phi', data=true_phase_maps[1:, ...])
  dataset.close()    