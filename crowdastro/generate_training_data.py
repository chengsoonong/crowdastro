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
    swire = f_h5['/swire/cdfs/numeric']
    fluxes = swire[:, 2:7]
    stellarities = swire[:, 7]
    distances = swire[:, 8].reshape((-1, 1))
    images = swire[:, 8:]
    coords = swire[:, :2]

    # We now need to find the labels for each.
    truths = set(f_h5['/atlas/cdfs/consensus_objects'][:, 1])
    labels = numpy.array([o in truths for o in range(len(swire))])

    assert len(labels) == len(fluxes)
    assert len(fluxes) == len(stellarities)
    assert len(stellarities) == len(distances)
    assert len(distances) == len(images)

    features = numpy.hstack([fluxes, distances, images])

    n_astro = features.shape[1] - images.shape[1]

    # Save to HDF5.
    out_f_h5.create_dataset('labels', data=labels)
    out_f_h5.create_dataset('features', data=features)
    out_f_h5.create_dataset('positions', data=coords)

    # We want to ensure our training set is never in our testing set, so
    # 1. assign all ATLAS objects to a train or test set,
    # 2. if a SWIRE object is nearby a testing ATLAS object, assign it to a test
    #    set, and
    # 3. assign all other SWIRE objects to a train set.
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

    n_swire = len(fluxes)
    is_swire_train = numpy.ones((n_swire))
    is_swire_test = numpy.zeros((n_swire))

    for atlas_index in atlas_test_indices:
        swire = f_h5['/atlas/cdfs/numeric'][atlas_index, n_astro + IMAGE_SIZE:]
        nearby = (swire < ARCMIN).nonzero()[0]
        for swire_index in nearby:
            is_swire_test[swire_index] = 1
            is_swire_train[swire_index] = 0

    out_f_h5.create_dataset('is_atlas_train', data=is_atlas_train.astype(bool))
    out_f_h5.create_dataset('is_atlas_test', data=is_atlas_test.astype(bool))
    out_f_h5.create_dataset('is_swire_train', data=is_swire_train.astype(bool))
    out_f_h5.create_dataset('is_swire_test', data=is_swire_test.astype(bool))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='crowdastro.h5',
                        help='HDF5 input file')
    parser.add_argument('-o', default='training.h5',
                        help='HDF5 output file')
    args = parser.parse_args()

    with h5py.File(args.i, 'r') as f_h5:
        assert f_h5.attrs['version'] == '0.4.0'
        with h5py.File(args.o, 'w') as out_f_h5:
            generate(f_h5, out_f_h5)
