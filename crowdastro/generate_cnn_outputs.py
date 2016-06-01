"""Generates CNN output training data.

Matthew Alger
The Australian National University
2016
"""

import argparse
import csv
import logging

import h5py
import keras
import numpy

from .config import config

PATCH_RADIUS = config['patch_radius']
ARCMIN = 1 / 60


def generate(training_h5, cnn_model_json, cnn_weights_path):
    """Generates CNN output training data.

    training_h5: HDF5 file with training data.
    cnn_model_path: JSON model file.
    cnn_weights_path: Path to CNN weights HDF5 file.
    """
    cnn = keras.models.model_from_json(cnn_model_json.read())
    cnn.load_weights(cnn_weights_path)
    cnn.compile(loss='binary_crossentropy', optimizer='adadelta')
    get_convolutional_features_ = keras.backend.function(
            [cnn.layers[0].input], [cnn.layers[5].output])
    get_convolutional_features = (lambda p:
            get_convolutional_features_([p])[0].reshape((p.shape[0], -1)))

    images = training_h5['raw_patches'].value
    images = images.reshape((images.shape[0], 1, images.shape[1],
                             images.shape[2]))

    if 'cnn_outputs' in training_h5:
        del training_h5['cnn_outputs']

    out = training_h5.create_dataset('cnn_outputs', dtype=float,
                                     shape=(len(images), 32))

    batch_size = 1000
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        out[i : i + batch_size] = get_convolutional_features(batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='training.h5',
                        help='HDF5 training file')
    parser.add_argument('--model', default='model.json',
                        help='JSON CNN model')
    parser.add_argument('--weights', default='weights.h5',
                        help='HDF5 CNN weights')
    args = parser.parse_args()

    with h5py.File(args.training, 'r+') as training_h5:
        with open(args.model, 'r') as cnn_model_json:
            generate(training_h5, cnn_model_json, args.weights)
