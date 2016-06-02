"""Trains classifiers.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging

import h5py
import numpy
import sklearn.externals.joblib
import sklearn.ensemble
import sklearn.linear_model

from .config import config

PATCH_RADIUS = config['patch_radius']
PATCH_DIAMETER = PATCH_RADIUS * 2

def train(inputs_h5, training_h5, classifier_out_path, classifier='lr',
          use_astro=True, use_cnn=True):
    """Trains logistic regression.

    inputs_h5: Inputs HDF5 file.
    training_h5: Training HDF5 file.
    classifier_out_path: Output random forest path.
    classifier: Classifier to use in {'lr', 'rf'}. Default 'lr'.
    use_astro: Use astronomical features. Default True.
    use_cnn: Use CNN features. Default True.
    """
    if not any([use_astro, use_cnn]):
        raise ValueError('Must have features.')

    if classifier == 'lr':
        classifier = sklearn.linear_model.LogisticRegression(
                class_weight='balanced')
    elif classifier == 'rf':
        classifier = sklearn.ensemble.RandomForestClassifier(
                class_weight='balanced')
    else:
        raise ValueError('Unknown classifier: {}'.format(classifier))

    training_indices = (training_h5['sets'][:, 0] == 1).nonzero()[0]
    outputs = training_h5['labels'].value

    astro_inputs = training_h5['astro'].value
    image_inputs = training_h5['cnn_outputs'].value

    features = []
    if use_astro:
        features.append(astro_inputs)
    if use_cnn:
        features.append(image_inputs)
    inputs = numpy.hstack(features)

    inputs = inputs[training_indices]
    outputs = outputs[training_indices]

    classifier.fit(inputs, outputs)
    sklearn.externals.joblib.dump(classifier, classifier_out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default='crowdastro.h5',
                        help='HDF5 input data file')
    parser.add_argument('--training', default='training.h5',
                        help='HDF5 training data file')
    parser.add_argument('-o', default='classifier.pkl',
                        help='classifier output file')
    parser.add_argument('--classifier', choices={'lr', 'rf'}, default='lr',
                        help='which classifier to train')
    parser.add_argument('--no_astro', action='store_false', default=True,
                        help='ignore astro features')
    parser.add_argument('--no_cnn', action='store_false', default=True,
                        help='ignore CNN features')
    args = parser.parse_args()

    logging.root.setLevel(logging.DEBUG)

    with h5py.File(args.training, 'r') as training_h5:
        with h5py.File(args.inputs, 'r') as inputs_h5:
            train(inputs_h5, training_h5, args.o, classifier=args.classifier,
                  use_astro=args.no_astro, use_cnn=args.no_cnn)
