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
PADDING = 150

def get_patch(source, x, y, images_path):
  """Gets the image patch associated with x, y for the given source."""
  if source.startswith('C'):
      field = 'cdfs'
  else:
      field = 'elais'
  filename = '{}_{}.fits'.format(source, 'radio')
  path = os.path.join(images_path, field, '5x5', filename)
  # Pre-allocate.
  with astropy.io.fits.open(path, ignore_blank=True) as fits:
    radio_image = fits[0].data
    return radio_image[int(x - RADIUS + PADDING) : int(x + RADIUS + PADDING),
                       int(y - RADIUS + PADDING) : int(y + RADIUS + PADDING)]

def train(data_path, images_path, model_path, weights_path, epochs, batch_size):
  """Trains a CNN.

  data_path: Path to input data HDF5 file.
  images_path: Path to images directory containing cdfs/elais folders.
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
    xs = hf['data']['x']
    ys = hf['data']['y']
    labels = hf['data']['is_host']

  # Generate the training inputs. Each input is an image, so for each potential
  # host I'll have to fetch the subject image, slice it, and store it. This is
  # pretty big and pandas doesn't like it, otherwise I'd pregenerate it.

  # Allocate memory.
  n = len(xs)  # Number of examples.
  training_inputs = numpy.zeros(shape=(n, 1, RADIUS * 2, RADIUS * 2))
  training_outputs = numpy.zeros(shape=(n,))

  # Fill allocated memory.
  for index, (x, y, label, source) in enumerate(zip(xs, ys, labels, sources)):
    source = source.decode('ascii')
    patch = get_patch(source, x, y, images_path)
    training_inputs[index] = patch
    training_outputs[index] = label

  model.fit(training_inputs, training_outputs, batch_size=batch_size,
            nb_epoch=epochs)
  model.save_weights(weights_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data', help='path to training data HDF5')
  parser.add_argument('images', help='path to images directory')
  parser.add_argument('model', help='path to model JSON')
  parser.add_argument('weights', help='path to output weights HDF5')
  parser.add_argument('epochs', help='number of epochs', type=int)
  parser.add_argument('batch_size', help='batch size', type=int)
  args = parser.parse_args()

  train(args.data, args.images, args.model, args.weights, args.epochs,
        args.batch_size)