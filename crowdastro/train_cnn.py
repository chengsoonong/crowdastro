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


PATCH_RADIUS = config['patch_radius']
PATCH_DIAMETER = PATCH_RADIUS * 2


def train(training_h5, model_json, weights_path, epochs, batch_size):
    """Trains a CNN.

    training_h5: Training HDF5 file.
    model_json: JSON model file.
    weights_path: Output weights HDF5 file.
    epochs: Number of training epochs.
    batch_size: Batch size.
    """
    model = keras.models.model_from_json(model_json.read())
    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    training_inputs = training_h5['raw_patches'].value
    training_outputs = training_h5['labels'].value
    assert training_inputs.shape[0] == training_outputs.shape[0]

    sets = training_h5['sets']
    training_indices = sets[:, 0].nonzero()[0]
    training_inputs = training_inputs[training_indices]
    training_outputs = training_outputs[training_indices]

    zero_indices = (training_outputs == 0).nonzero()[0]
    one_indices = (training_outputs == 1).nonzero()[0]
    subset_zero_indices = numpy.random.choice(zero_indices,
            size=(len(one_indices,)), replace=False)
    all_indices = numpy.hstack([subset_zero_indices, one_indices])
    all_indices.sort()

    training_inputs = training_inputs[all_indices]
    training_outputs = training_outputs[all_indices]
    assert (training_outputs == 1).sum() == (training_outputs == 0).sum()

    training_inputs = training_inputs.reshape((
            training_inputs.shape[0], 1, training_inputs.shape[1],
            training_inputs.shape[2]))

    model.fit(training_inputs, training_outputs, batch_size=batch_size,
              nb_epoch=epochs)
    model.save_weights(weights_path, overwrite=True)

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
        with open(args.model, 'r') as model_json:
            train(training_h5, model_json, args.output, int(args.epochs),
                  int(args.batch_size))
