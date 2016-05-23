#!/usr/bin/env python3

"""Tests a convolutional neural network model.

This is a self-contained module.

Usage:
  ./test_cnn.py data_path model_path weights_path

Matthew Alger
The Australian National University
2016
"""

import argparse
import os.path

import astropy.io.fits
import h5py
import keras.models
import numpy

RADIUS = 40

def test(data_path, model_path, weights_path, batch_size):
  """Trains a CNN.

  data_path: Path to input data HDF5 file.
  model_path: Path to JSON model.
  weights_path: Path to weights HDF5 file.
  batch_size: Batch size.
  """
  with open(model_path) as f:
    model = keras.models.model_from_json(f.read())
  model.compile(loss='binary_crossentropy', optimizer='adadelta')

  with h5py.File(data_path) as hf:
    sources = hf['data']['source']
    testing_inputs = hf['inputs']
    testing_outputs = hf['outputs']

    n = testing_inputs.shape[0] // 2  # Number of examples, 0.5 train/test.
    testing_inputs = testing_inputs[n:]  # Second half for testing.
    testing_outputs = testing_outputs[n:]
    print(model.evaluate(testing_inputs, testing_outputs))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data', help='path to data HDF5')
  parser.add_argument('model', help='path to model JSON')
  parser.add_argument('weights', help='path to weights HDF5')
  parser.add_argument('batch_size', help='batch size', type=int)
  args = parser.parse_args()

  test(args.data, args.model, args.weights, args.batch_size)
