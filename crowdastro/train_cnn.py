"""Trains a convolutional neural network.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging
import os.path

import astropy.io.fits
import h5py
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
    import keras.models
    model = keras.models.model_from_json(model_json.read())
    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    train_set = training_h5['is_ir_train'].value
    ir_survey = training_h5.attrs['ir_survey']

    if ir_survey == 'swire':
        training_inputs = training_h5['features'].value[train_set, 6:]
    elif ir_survey == 'wise':
        training_inputs = training_h5['features'].value[train_set, 5:]

    training_inputs = training_inputs.reshape(
            (-1, 1, PATCH_DIAMETER, PATCH_DIAMETER))
    training_outputs = training_h5['labels'].value[train_set]
    assert training_inputs.shape[0] == training_outputs.shape[0]

    # Downsample for class balance.
    zero_indices = (training_outputs == 0).nonzero()[0]
    one_indices = (training_outputs == 1).nonzero()[0]
    subset_zero_indices = numpy.random.choice(zero_indices,
            size=(len(one_indices,)), replace=False)
    all_indices = numpy.hstack([subset_zero_indices, one_indices])
    all_indices.sort()

    training_inputs = training_inputs[all_indices]
    training_outputs = training_outputs[all_indices]
    assert (training_outputs == 1).sum() == (training_outputs == 0).sum()

    model.fit(training_inputs, training_outputs, batch_size=batch_size,
              nb_epoch=epochs)
    model.save_weights(weights_path, overwrite=True)


def _populate_parser(parser):
    parser.description = 'Trains a convolutional neural network.'
    parser.add_argument('--h5', default='data/training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--model', default='data/model.json',
                        help='JSON model file')
    parser.add_argument('--output', default='data/weights.h5',
                        help='HDF5 file for output weights')
    parser.add_argument('--epochs', default=10,
                        help='number of epochs to train for')
    parser.add_argument('--batch_size', default=100, help='batch size')


def _main(args):
    with h5py.File(args.h5, 'r') as training_h5:
        with open(args.model, 'r') as model_json:
            train(training_h5, model_json, args.output, int(args.epochs),
                  int(args.batch_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
