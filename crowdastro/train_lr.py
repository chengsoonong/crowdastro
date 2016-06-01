"""Trains logistic regression.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging
import os.path

import astropy.io.fits
import h5py
import keras.models
import numpy


from .config import config


PATCH_RADIUS = config['patch_radius']
PATCH_DIAMETER = PATCH_RADIUS * 2


def train(training_h5, model_json, weights_path, epochs, batch_size):
    """Trains a CNN.

    training_h5: Training HDF5 file.
    model_json: JSON model file.
    weights_path: CNN weights HDF5 file.
    epochs: Number of training epochs.
    batch_size: Batch size.
    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', default='training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--cnn_model', default='model.json',
                        help='JSON CNN model file')
    parser.add_argument('--cnn_weights', default='weights.h5',
                        help='HDF5 file for CNN weights')
    args = parser.parse_args()

    logging.root.setLevel(logging.DEBUG)

    with h5py.File(args.h5, 'r') as training_h5:
        with open(args.model, 'r') as model_json:
            train(training_h5, model_json, args.output, int(args.epochs),
                  int(args.batch_size))
