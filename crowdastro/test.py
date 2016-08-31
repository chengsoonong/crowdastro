"""Tests classifiers on radio subjects.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging

import h5py
import numpy
import sklearn.externals.joblib
import sklearn.neighbors

from .config import config

PATCH_RADIUS = config['patch_radius']
PATCH_DIAMETER = PATCH_RADIUS * 2
ARCMIN = 1 / 60
FITS_HEIGHT = config['surveys']['atlas']['fits_height']
FITS_WIDTH = config['surveys']['atlas']['fits_height']
IMAGE_SIZE = FITS_HEIGHT * FITS_WIDTH


def test(inputs_h5, training_h5, classifier_path, astro_transformer_path,
         image_transformer_path, use_astro=True, use_cnn=True):
    """Tests classifiers.

    inputs_h5: Inputs HDF5 file.
    training_h5: Training HDF5 file.
    classifier_path: Path to classifier.
    astro_transformer_path: Path to astro transformer.
    image_transformer_path: Path to image transformer.
    """
    # TODO(MatthewJA): Figure out a neat way to test on subjects stored in
    # training_h5.
    classifier = sklearn.externals.joblib.load(classifier_path)
    astro_transformer = sklearn.externals.joblib.load(astro_transformer_path)
    image_transformer = sklearn.externals.joblib.load(image_transformer_path)

    assert training_h5.attrs['ir_survey'] == inputs_h5.attrs['ir_survey']
    n_static = 6 if training_h5.attrs['ir_survey'] == 'swire' else 5

    test_indices = training_h5['is_atlas_test'].value
    numeric_subjects = inputs_h5['/atlas/cdfs/numeric'][test_indices, :]

    n_correct = 0
    n_total = 0
    for subject in numeric_subjects:
        swire = subject[2 + IMAGE_SIZE:]
        nearby = swire < ARCMIN
        astro_inputs = numpy.minimum(training_h5['features'][nearby, :n_static],
                                     1500)
        image_inputs = training_h5['features'][nearby, n_static:]

        features = []
        if use_astro:
            features.append(astro_transformer.fit_transform(astro_inputs))
        if use_cnn:
            features.append(image_transformer.fit_transform(image_inputs))
        inputs = numpy.hstack(features)

        outputs = training_h5['labels'][nearby]

        selection = classifier.predict_proba(inputs)[:, 1].argmax()
        n_correct += outputs[selection]
        n_total += 1

    print('{:.02%}'.format(n_correct / n_total))


def _populate_parser(parser):
    parser.description = 'Tests classifiers on radio subjects.'
    parser.add_argument('--inputs', default='data/crowdastro.h5',
                        help='HDF5 crowdastro data file')
    parser.add_argument('--training', default='data/training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--classifier', default='data/classifier.pkl',
                        help='classifier file')
    parser.add_argument('--astro_transformer',
                        default='data/astro_transformer.pkl',
                        help='astro transformer file')
    parser.add_argument('--image_transformer',
                        default='data/image_transformer.pkl',
                        help='image transformer file')
    parser.add_argument('--no_astro', action='store_false', default=True,
                        help='ignore astro features')
    parser.add_argument('--no_cnn', action='store_false', default=True,
                        help='ignore CNN features')


def _main(args):
    with h5py.File(args.training, 'r') as training_h5:
        with h5py.File(args.inputs, 'r') as inputs_h5:
            assert inputs_h5.attrs['version'] == '0.5.0'
            test(inputs_h5, training_h5, args.classifier,
                 args.astro_transformer, args.image_transformer,
                 use_astro=args.no_astro, use_cnn=args.no_cnn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
