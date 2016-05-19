#!/usr/bin/env python3

"""Trains a convolutional neural network model.

This is a self-contained module.

Usage:
  ./train_cnn.py in_data_path images_path in_model_path out_weights_path epochs

Matthew Alger
The Australian National University
2016
"""

# This module should have minimal imports, so we can easily run it on GPU
# clusters.

import argparse
import os.path

import astropy.io.fits
import h5py
import keras.models
import numpy

RADIUS = 40

def train(data_path, model_path, weights_path, epochs, batch_size):
  """Trains a CNN.

  data_path: Path to input data HDF5 file.
  model_path: Path to JSON model.
  weights_path: Path to output weights HDF5 file.
  epochs: Number of training epochs.
  batch_size: Batch size.
  """
  with open(model_path) as f:
    model = keras.models.model_from_json(f.read())
  model.compile(loss='binary_crossentropy', optimizer='adadelta')

  with h5py.File(data_path) as hf:
    sources = hf['data']['source']
    training_inputs = hf['inputs']
    training_outputs = hf['outputs']

    # Generate the training inputs. Each input is an image, so for each
    # potential host I'll have to fetch the subject image, slice it, and store
    # it. This is pretty big and pandas doesn't like it, otherwise I'd
    # pregenerate it.

    n = training_inputs.shape[0] // 2  # Number of examples, 0.5 train/test.
    training_inputs = training_inputs[:n]
    training_outputs = training_outputs[:n]
    model.fit(training_inputs, training_outputs, batch_size=batch_size,
              nb_epoch=epochs)
    model.save_weights(weights_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data', help='path to training data HDF5')
  parser.add_argument('model', help='path to model JSON')
  parser.add_argument('weights', help='path to output weights HDF5')
  parser.add_argument('epochs', help='number of epochs', type=int)
  parser.add_argument('batch_size', help='batch size', type=int)
  args = parser.parse_args()

  train(args.data, args.model, args.weights, args.epochs, args.batch_size)