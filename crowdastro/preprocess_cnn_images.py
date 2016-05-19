#!/usr/bin/env python3

"""Generates and stores the training data for the CNN.

This is a self-contained module.

Usage:
  ./preprocess_cnn_images.py data_path images_path

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

def preprocess(data_path, images_path):
  """Preprocesses images.

  data_path: Path to input data HDF5 file.
  images_path: Path to images directory containing cdfs/elais folders.
  out_path: Path to output HDF5 file.
  """
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

  with h5py.File(data_path, 'r+') as hf:
    hf['inputs'] = training_inputs
    hf['outputs'] = training_outputs

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('data', help='path to training data HDF5')
  parser.add_argument('images', help='path to images directory')
  args = parser.parse_args()

  preprocess(args.data, args.images)