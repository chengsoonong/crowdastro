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

PATCH_RADIUS = config['patch_radius']  # px
ARCMIN = 1 / 60  # deg


def generate(training_h5, cnn_model_json, cnn_weights_path):
    """Generates CNN output training data.

    training_h5: HDF5 file with training data.
    cnn_model_path: JSON model file.
    cnn_weights_path: Path to CNN weights HDF5 file.
    """
    n_static = 6 if training_h5.attrs['ir_survey'] == 'swire' else 7
    cnn = keras.models.model_from_json(cnn_model_json.read())
    cnn.load_weights(cnn_weights_path)
    cnn.compile(loss='binary_crossentropy', optimizer='adadelta')
    get_convolutional_features_ = keras.backend.function(
            [cnn.layers[0].input], [cnn.layers[5].output])
    get_convolutional_features = (lambda p:
            get_convolutional_features_([p])[0].reshape((p.shape[0], -1)))

    images = training_h5['features'][:, n_static:].reshape(
            (-1, 1, PATCH_RADIUS * 2, PATCH_RADIUS * 2))

    test_out = get_convolutional_features(images[:1, :, :, :])

    if '_features' in training_h5:
        del training_h5['_features']

    out = training_h5.create_dataset('_features', dtype=float,
            shape=(len(images), n_static + test_out.shape[1]))

    # Copy the static features across. We'll fill in the rest with the CNN.
    out[:, :n_static] = training_h5['features'][:, :n_static]

    batch_size = 1000
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        out[i : i + batch_size, n_static:] = get_convolutional_features(batch)

    # Clean up - delete the original features and rename our new features.
    del training_h5['features']
    training_h5['features'] = training_h5['_features']
    del training_h5['_features']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', default='data/training.h5',
                        help='HDF5 training file')
    parser.add_argument('--model', default='data/model.json',
                        help='JSON CNN model')
    parser.add_argument('--weights', default='data/weights.h5',
                        help='HDF5 CNN weights')
    args = parser.parse_args()

    with h5py.File(args.training, 'r+') as training_h5:
        with open(args.model, 'r') as cnn_model_json:
            generate(training_h5, cnn_model_json, args.weights)
