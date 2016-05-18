#!/usr/bin/env python3

"""Trains a convolutional neural network model.

This is a self-contained module.

Usage:
  ./train_cnn.py in_data_path images_path in_model_path out_weights_path

Matthew Alger
The Australian National University
2016
"""

# This module should have minimal imports, so we can easily run it on GPU
# clusters.

import argparse
import os.path

import astropy.io
import keras.models
import numpy
import pandas
import sklearn.cross_validation

RADIUS = 40
PADDING = 150

def get_patch(source, x, y, images_path):
  """Gets the image patch associated with x, y for the given source."""
  if source.startswith('C'):
      field = 'cdfs'
  else:
      field = 'elais'
  filename = '{}_{}.fits'.format(source, wavelength)
  path = os.path.join(images_path, field, '5x5', filename)
  with astropy.io.fits.open(path, ignore_blank=True) as fits:
    radio_image = fits_file[0].data
    return radio_image[int(x - RADIUS + PADDING) : int(x + RADIUS + PADDING),
                       int(y - RADIUS + PADDING) : int(y + RADIUS + PADDING)]

def train(data_path, images_path, model_path, weights_path):
  """Trains a CNN.

  data_path: Path to input data HDF5 file.
  images_path: Path to images directory containing cdfs/elais folders.
  model_path: Path to JSON model.
  weights_path: Path to output weights HDF5 file.
  """
  with open(model_path) as f:
    model = keras.models.model_from_json(f.read())

  with pandas.HDFStore(data_path) as store:
    dataframe = store['data']
    sources = dataframe['sources']
    xs = dataframe['x'].as_matrix()
    ys = dataframe['y'].as_matrix()
    labels = dataframe['is_host'].as_matrix()

  # Generate the training inputs. Each input is an image, so for each potential
  # host I'll have to fetch the subject image, slice it, and store it. This is
  # pretty big and pandas doesn't like it, otherwise I'd pregenerate it.
  training_inputs = []
  training_outputs = []
  for x, y, label, (source,) in zip(xs, ys, labels, sources.iterrows()):
    patch = get_patch(source, x, y, images_path)
    training_inputs.append(patch)
    training_outputs.append(label)
  training_inputs = numpy.array(training_inputs)
  training_outputs = numpy.array(training_outputs)

  im_size = training_inputs.shape[1:]
  training_inputs = training_inputs.reshape(training_inputs.shape[0], 1,
                                            im_size[0], im_size[1])

  model.fit(training_inputs, training_outputs)
  model.save_weights(weights_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data', help='path to training data HDF5')
  parser.add_argument('images', help='path to images directory')
  parser.add_argument('model', help='path to model JSON')
  parser.add_argument('weights', help='path to output weights HDF5')
  args = parser.parse_args()

  train(args.data, args.images, args.model, args.weights)