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
import sklearn.neighbors

from .config import config

PATCH_RADIUS = config['patch_radius']
PATCH_DIAMETER = PATCH_RADIUS * 2
ARCMIN = 1 / 60

def test(inputs_h5, training_h5, classifier_path, astro_transformer_path,
         image_transformer_path, use_astro=True, use_cnn=True, simple=False):
    """Tests classifiers.

    inputs_h5: Inputs HDF5 file.
    training_h5: Training HDF5 file.
    classifier_path: Path to classifier.
    astro_transformer_path: Path to astro transformer.
    image_transformer_path: Path to image transformer.
    simple: Test only on subjects containing one radio source. Default False.
    """
    # TODO(MatthewJA): Figure out a neat way to test on subjects stored in
    # training_h5.
    classifier = sklearn.externals.joblib.load(classifier_path)
    astro_transformer = sklearn.externals.joblib.load(astro_transformer_path)
    image_transformer = sklearn.externals.joblib.load(image_transformer_path)

    testing_indices = inputs_h5['/atlas/cdfs/testing_indices'].value
    swire_positions = inputs_h5['/swire/cdfs/catalogue'][:, :2]
    atlas_positions = inputs_h5['/atlas/cdfs/positions'].value
    all_astro_inputs = training_h5['astro'].value
    all_cnn_inputs = training_h5['cnn_outputs'].value
    all_labels = training_h5['labels'].value

    swire_tree = sklearn.neighbors.KDTree(swire_positions, metric='chebyshev')

    if simple:
        atlas_counts = {}  # ATLAS ID to number of objects in that subject.
        for consensus in inputs_h5['/atlas/cdfs/consensus_objects']:
            atlas_id = int(consensus[0])
            atlas_counts[atlas_id] = atlas_counts.get(atlas_id, 0) + 1

        indices = []
        for atlas_id, count in atlas_counts.items():
            if count == 1 and atlas_id in testing_indices:
                indices.append(atlas_id)

        indices = numpy.array(sorted(indices))

        atlas_positions = atlas_positions[indices]
        logging.debug('Found %d simple subjects.', len(atlas_positions))
    else:
        atlas_positions = atlas_positions[testing_indices]
        logging.debug('Found %d subjects.', len(atlas_positions))

    # Test each ATLAS subject.
    n_correct = 0
    n_total = 0
    for pos in atlas_positions:
        neighbours, distances = swire_tree.query_radius([pos], ARCMIN,
                                                        return_distance=True)
        neighbours = neighbours[0]
        distances = distances[0]

        astro_inputs = all_astro_inputs[neighbours]
        astro_inputs[:, -1] = distances
        cnn_inputs = all_cnn_inputs[neighbours]
        labels = all_labels[neighbours]

        features = []
        if use_astro:
            features.append(astro_transformer.transform(astro_inputs))
        if use_cnn:
            features.append(image_transformer.transform(cnn_inputs))
        inputs = numpy.hstack(features)
        outputs = classifier.predict_proba(inputs)[:, 1]
        assert len(labels) == len(outputs)
        index = outputs.argmax()
        n_correct += labels[index]
        n_total += 1
    print('{:.02%}'.format(n_correct / n_total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', default='crowdastro.h5',
                        help='HDF5 inputs data file')
    parser.add_argument('--training', default='training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--classifier', default='classifier.pkl',
                        help='classifier file')
    parser.add_argument('--astro_transformer', default='astro_transformer.pkl',
                        help='astro transformer file')
    parser.add_argument('--image_transformer', default='image_transformer.pkl',
                        help='image transformer file')
    parser.add_argument('--no_astro', action='store_false', default=True,
                        help='ignore astro features')
    parser.add_argument('--no_cnn', action='store_false', default=True,
                        help='ignore CNN features')
    parser.add_argument('--simple', action='store_true', default=False,
                        help='use only single-AGN subjects')
    args = parser.parse_args()

    logging.getLogger('sknn.mlp').setLevel(logging.WARNING)

    with h5py.File(args.training, 'r') as training_h5:
        with h5py.File(args.inputs, 'r') as inputs_h5:
            test(inputs_h5, training_h5, args.classifier,
                 args.astro_transformer, args.image_transformer,
                 use_astro=args.no_astro, use_cnn=args.no_cnn,
                 simple=args.simple)
