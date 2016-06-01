"""Tests logistic regression on subjects.

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
import sklearn.neighbors

from .config import config


PATCH_RADIUS = config['patch_radius']
PATCH_DIAMETER = PATCH_RADIUS * 2
ARCMIN = 1 / 60


def test(inputs_h5, training_h5, lr_path):
    """Trains logistic regression.

    inputs_h5: Inputs HDF5 file.
    training_h5: Training HDF5 file.
    lr_path: Path to logistic regression.
    """
    # TODO(MatthewJA): Figure out a neat way to test on subjects stored in
    # training_h5.
    lr = sklearn.externals.joblib.load(lr_path)

    testing_indices = inputs_h5['/atlas/cdfs/testing_indices'].value
    swire_positions = inputs_h5['/swire/cdfs/catalogue'][:, :2]
    swire_tree = sklearn.neighbors.KDTree(swire_positions, metric='manhattan')
    atlas_positions = inputs_h5['/atlas/cdfs/positions'].value[testing_indices]
    all_astro_inputs = training_h5['astro'].value
    all_cnn_inputs = training_h5['cnn_outputs'].value
    all_labels = training_h5['labels'].value

    # Test each ATLAS subject.
    n_correct = 0
    n_total = 0
    for pos in atlas_positions:
        neighbours = swire_tree.query_radius([pos], ARCMIN)[0]
        astro_inputs = all_astro_inputs[neighbours]
        cnn_inputs = all_cnn_inputs[neighbours]
        labels = all_labels[neighbours]
        inputs = numpy.hstack([astro_inputs, cnn_inputs])

        outputs = lr.predict_proba(inputs)[:, 1]
        assert len(labels) == len(outputs)
        index = outputs.argmax()
        n_correct += labels[index]
        n_total += 1
        print('{:.02%}'.format(n_correct / n_total), end='\r')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default='crowdastro.h5',
                        help='HDF5 inputs data file')
    parser.add_argument('--training', default='training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--lr', default='lr.pkl',
                        help='Logistic regression file')
    args = parser.parse_args()

    logging.root.setLevel(logging.DEBUG)

    with h5py.File(args.training, 'r') as training_h5:
        with h5py.File(args.inputs, 'r') as inputs_h5:
            test(inputs_h5, training_h5, args.lr)
