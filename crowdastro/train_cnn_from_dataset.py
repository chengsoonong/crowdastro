"""Trains a convolutional neural network from dataset.h5.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging

import h5py
import numpy

from .config import config

PATCH_RADIUS = config['patch_radius']
PATCH_DIAMETER = PATCH_RADIUS * 2


def train(dataset_h5, model_json, weights_path, epochs, batch_size):
    """Trains a CNN.

    training_h5: Dataset HDF5 file.
    model_json: JSON model file.
    weights_path: Output weights HDF5 file.
    epochs: Number of training epochs.
    batch_size: Batch size.
    """
    import keras.callbacks
    import keras.models
    model = keras.models.model_from_json(model_json.read())
    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    n_nonimage_features = 5
    training_inputs = dataset_h5['features'].value[:, n_nonimage_features:]
    training_inputs = training_inputs.reshape(
            (-1, 1, PATCH_DIAMETER, PATCH_DIAMETER))
    training_outputs = dataset_h5['labels'].value
    assert training_inputs.shape[0] == training_outputs.shape[0]

    # Downsample for class balance.
    zero_indices = (training_outputs == 0).nonzero()[0]
    one_indices = (training_outputs == 1).nonzero()[0]
    subset_zero_indices = numpy.random.choice(
        zero_indices, size=(len(one_indices,)), replace=False)
    all_indices = numpy.hstack([subset_zero_indices, one_indices])
    all_indices.sort()

    training_inputs = training_inputs[all_indices]
    training_outputs = training_outputs[all_indices]
    assert (training_outputs == 1).sum() == (training_outputs == 0).sum()

    try:
        model.load_weights(weights_path)
        logging.info('Loaded weights.')
    except OSError:
        logging.warning('Couldn\'t load weights file. Creating new file...')
        pass

    # TODO(MatthewJA): Clean this up!
    callbacks = [
        keras.callbacks.ModelCheckpoint(
                'weights_progress.h5',
                monitor='val_loss',
                verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode='auto')
    ]
    model.fit(training_inputs, training_outputs, batch_size=batch_size,
              nb_epoch=epochs, callbacks=callbacks)
    model.save_weights(weights_path, overwrite=True)


def _populate_parser(parser):
    parser.description = 'Trains a convolutional neural network.'
    parser.add_argument('--h5', default='data/dataset.h5',
                        help='HDF5 dataset file')
    parser.add_argument('--model', default='data/model.json',
                        help='JSON model file')
    parser.add_argument('--output', default='data/weights.h5',
                        help='HDF5 file for output weights')
    parser.add_argument('--epochs', default=10,
                        help='number of epochs to train for')
    parser.add_argument('--batch_size', default=100, help='batch size')


def _main(args):
    with h5py.File(args.h5, 'r') as dataset_h5:
        with open(args.model, 'r') as model_json:
            train(dataset_h5, model_json, args.output, int(args.epochs),
                  int(args.batch_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
