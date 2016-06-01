"""Trains logistic regression.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging

import h5py
import numpy
import sklearn.externals.joblib
import sklearn.linear_model

from .config import config


PATCH_RADIUS = config['patch_radius']
PATCH_DIAMETER = PATCH_RADIUS * 2


def train(training_h5, lr_out_path):
    """Trains logistic regression.

    training_h5: Training HDF5 file.
    lr_out_path: Output logistic regression path.
    """
    lr = sklearn.linear_model.LogisticRegression(class_weight='balanced')
    training_indices = (training_h5['sets'][:, 0] == 1).nonzero()[0]
    astro_inputs = training_h5['astro'].value
    image_inputs = training_h5['cnn_outputs'].value
    outputs = training_h5['labels'].value
    inputs = numpy.hstack([astro_inputs, image_inputs])

    inputs = inputs[training_indices]
    outputs = outputs[training_indices]

    lr.fit(inputs, outputs)
    sklearn.externals.joblib.dump(lr, lr_out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='training.h5',
                        help='HDF5 training data file')
    parser.add_argument('-o', default='lr.pkl',
                        help='logistic regression output file')
    args = parser.parse_args()

    logging.root.setLevel(logging.DEBUG)

    with h5py.File(args.i, 'r') as training_h5:
        train(training_h5, args.o)
