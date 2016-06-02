"""Tests classifiers on subjects.

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

def test(inputs_h5, training_h5, classifier_path, use_astro=True, use_cnn=True):
    """Tests classifiers.

    inputs_h5: Inputs HDF5 file.
    training_h5: Training HDF5 file.
    classifier_path: Path to classifier.
    """
    # TODO(MatthewJA): Figure out a neat way to test on subjects stored in
    # training_h5.
    classifier = sklearn.externals.joblib.load(classifier_path)

    testing_indices = inputs_h5['/atlas/cdfs/testing_indices'].value
    swire_positions = inputs_h5['/swire/cdfs/catalogue'][:, :2]
    atlas_positions = inputs_h5['/atlas/cdfs/positions'].value[testing_indices]
    all_astro_inputs = training_h5['astro'].value
    all_cnn_inputs = training_h5['cnn_outputs'].value
    all_labels = training_h5['labels'].value

    swire_tree = sklearn.neighbors.KDTree(swire_positions, metric='chebyshev')

    # Test each ATLAS subject.
    n_correct = 0
    n_total = 0
    for pos in atlas_positions:
        neighbours = swire_tree.query_radius([pos], ARCMIN)[0]
        astro_inputs = all_astro_inputs[neighbours]
        cnn_inputs = all_cnn_inputs[neighbours]
        labels = all_labels[neighbours]

        features = []
        if use_astro:
            features.append(astro_inputs)
        if use_cnn:
            features.append(cnn_inputs)
        inputs = numpy.hstack(features)

        outputs = classifier.predict_proba(inputs)[:, 1]
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
    parser.add_argument('--classifier', default='classifier.pkl',
                        help='classifier file')
    parser.add_argument('--no_astro', action='store_false', default=True,
                        help='ignore astro features')
    parser.add_argument('--no_cnn', action='store_false', default=True,
                        help='ignore CNN features')
    args = parser.parse_args()

    logging.root.setLevel(logging.DEBUG)

    with h5py.File(args.training, 'r') as training_h5:
        with h5py.File(args.inputs, 'r') as inputs_h5:
            test(inputs_h5, training_h5, args.classifier,
                 use_astro=args.no_astro, use_cnn=args.no_cnn)
