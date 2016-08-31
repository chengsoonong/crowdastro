"""Generates training data (potential hosts and their astronomical features).

Also generates training/testing indices.

Matthew Alger
The Australian National University
2016
"""

import argparse

import h5py
import numpy
import sklearn.neighbors

from .config import config

FITS_HEIGHT = config['surveys']['atlas']['fits_height']
FITS_WIDTH = config['surveys']['atlas']['fits_height']
IMAGE_SIZE = FITS_HEIGHT * FITS_WIDTH  # px
ARCMIN = 1 / 60  # deg


def remove_nans(n):
    """Replaces NaN with 0."""
    if numpy.ma.is_masked(n):
        return 0

    return float(n)


def generate(f_h5, out_f_h5):
    """Generates potential hosts and their astronomical features.

    f_h5: crowdastro input HDF5 file.
    out_f_h5: Training data output HDF5 file.
    """
    if f_h5.attrs['ir_survey'] == 'swire':
        swire = f_h5['/swire/cdfs/numeric']
        astro_features = swire[:, 2:7]
        distances = swire[:, 7].reshape((-1, 1))
        images = swire[:, 8:]
        coords = swire[:, :2]
    elif f_h5.attrs['ir_survey'] == 'wise':
        wise = f_h5['/wise/cdfs/numeric']
        magnitudes = wise[:, 2:6]
        distances = wise[:, 6].reshape((-1, 1))
        images = wise[:, 7:]
        coords = wise[:, :2]

        # Magnitude differences are probably useful features.
        w1_w2 = magnitudes[:, 0] - magnitudes[:, 1]
        w2_w3 = magnitudes[:, 1] - magnitudes[:, 2]

        # Converting the magnitudes to a linear scale seems to improve
        # performance.
        linearised_magnitudes = numpy.power(10, -0.4 * magnitudes)
        w1_w2 = numpy.power(10, -0.4 * w1_w2)
        w2_w3 = numpy.power(10, -0.4 * w2_w3)
        astro_features = numpy.concatenate([linearised_magnitudes,
                w1_w2.reshape((-1, 1)),
                w2_w3.reshape((-1, 1))], axis=1)


    # We now need to find the labels for each.
    truths = set(f_h5['/atlas/cdfs/consensus_objects'][:, 1])
    labels = numpy.array([o in truths for o in range(len(astro_features))])

    assert len(labels) == len(astro_features)
    assert len(astro_features) == len(distances)
    assert len(distances) == len(images)

    features = numpy.hstack([astro_features, distances, images])
    n_astro = features.shape[1] - images.shape[1]

    # Save to HDF5.
    out_f_h5.create_dataset('labels', data=labels)
    out_f_h5.create_dataset('features', data=features)
    out_f_h5.create_dataset('positions', data=coords)
    out_f_h5.attrs['ir_survey'] = f_h5.attrs['ir_survey']

    # These testing sets are used for training the CNN. For training/testing
    # models, use crowdastro.generate_test_sets.

    # We want to ensure our training set is never in our testing set, so
    # 1. assign all ATLAS objects to a train or test set,
    # 2. if a IR object is nearby a testing ATLAS object, assign it to a test
    #    set, and
    # 3. assign all other IR objects to a train set.
    n_atlas = f_h5['/atlas/cdfs/numeric'].shape[0]
    indices = numpy.arange(n_atlas)
    numpy.random.shuffle(indices)
    atlas_test_indices = indices[:int(n_atlas * config['test_size'])]
    atlas_train_indices = indices[int(n_atlas * config['test_size']):]

    atlas_test_indices.sort()
    atlas_train_indices.sort()

    is_atlas_train = numpy.zeros((n_atlas,))
    is_atlas_test = numpy.zeros((n_atlas,))

    is_atlas_test[atlas_test_indices] = 1
    is_atlas_train[atlas_train_indices] = 1

    n_ir = len(astro_features)
    is_ir_train = numpy.ones((n_ir))
    is_ir_test = numpy.zeros((n_ir))

    for atlas_index in atlas_test_indices:
        ir = f_h5['/atlas/cdfs/numeric'][atlas_index, n_astro + IMAGE_SIZE:]
        nearby = (ir < ARCMIN).nonzero()[0]
        for ir_index in nearby:
            is_ir_test[ir_index] = 1
            is_ir_train[ir_index] = 0

    out_f_h5.create_dataset('is_atlas_train', data=is_atlas_train.astype(bool))
    out_f_h5.create_dataset('is_atlas_test', data=is_atlas_test.astype(bool))
    out_f_h5.create_dataset('is_ir_train', data=is_ir_train.astype(bool))
    out_f_h5.create_dataset('is_ir_test', data=is_ir_test.astype(bool))


def _populate_parser(parser):
    parser.description = 'Generates training data (potential hosts and their ' \
                         'astronomical features).'
    parser.add_argument('-i', default='data/crowdastro.h5',
                        help='HDF5 input file')
    parser.add_argument('-o', default='data/training.h5',
                        help='HDF5 output file')


def _main(args):
    with h5py.File(args.i, 'r') as f_h5:
        assert f_h5.attrs['version'] == '0.5.1'
        with h5py.File(args.o, 'w') as out_f_h5:
            generate(f_h5, out_f_h5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
