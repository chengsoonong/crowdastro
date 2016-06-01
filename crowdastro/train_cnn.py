"""Trains a convolutional neural network model.

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


PATCH_RADIUS = 40


def train(training_h5, model_json, weights_h5, epochs, batch_size):
    """Trains a CNN.

    training_h5: Training HDF5 file.
    model_json: JSON model file.
    weights_h5: Output weights HDF5 file.
    epochs: Number of training epochs.
    batch_size: Batch size.
    """
    model = keras.models.model_from_json(model_json.read())
    # model.compile(loss='binary_crossentropy', optimizer='adadelta')

    training_inputs = training_h5['raw_patches']
    training_outputs = training_h5['labels']

    # Screen empty inputs/outputs.
    # TODO(MatthewJA): Pre-screen this in another part of the pipeline.
    nonzero = training_inputs.value[:10] != 0
    all_nonzero = numpy.apply_along_axis(numpy.any, 0, nonzero)
    print(all_nonzero)
    raise

    n = training_inputs.shape[0] // 2  # Number of examples, 0.5 train/test.
    training_inputs = training_inputs[:n]
    training_outputs = training_outputs[:n]
    model.fit(training_inputs, training_outputs, batch_size=batch_size,
              nb_epoch=epochs)
    model.save_weights(weights_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', default='training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--model', default='model.json',
                        help='JSON model file')
    parser.add_argument('--output', default='weights.h5',
                        help='HDF5 file for output weights')
    parser.add_argument('--epochs', default=10,
                        help='number of epochs to train for')
    parser.add_argument('--batch_size', default=100, help='batch size')
    args = parser.parse_args()

    logging.root.setLevel(logging.DEBUG)

    with h5py.File(args.h5, 'r') as training_h5:
        with h5py.File(args.output, 'w') as weights_h5:
            with open(args.model, 'r') as model_json:
                train(training_h5, model_json, weights_h5, args.epochs,
                      args.batch_size)
