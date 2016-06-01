"""Generates training data (potential hosts and their astronomical features).

Note that images are not included yet. These are processed separately.

Matthew Alger
The Australian National University
2016
"""

import argparse

import h5py
import numpy

from .config import config


def remove_nans(n):
    """Replaces NaN with 0."""
    if numpy.ma.is_masked(n):
        return 0

    return float(n)


def generate(f_h5, out_f_h5, simple=False):
    """Generates potential hosts and their astronomical features.

    f_h5: crowdastro input HDF5 file.
    out_f_h5: Training data output HDF5 file.
    simple: Optional. Whether to only use simple (one-host) subjects.
    """
    swire = f_h5['/swire/cdfs/catalogue']
    fluxes = swire[:, 2:7]
    stellarities = swire[:, 7]
    coords = swire[:, :2]

    # We now need to find the labels for each.
    truths = set(f_h5['/atlas/cdfs/consensus_objects'][:, 1])
    labels = numpy.array([o in truths for o in range(len(swire))])

    assert len(labels) == len(fluxes)
    assert len(fluxes) == len(stellarities)

    # Save to HDF5.
    out_f_h5.create_dataset('labels', data=labels)
    out_f_h5.create_dataset('astro', data=fluxes)
    out_f_h5.create_dataset('positions', data=coords)
    indices = out_f_h5.create_group('indices')
    indices.create_dataset('training',
                           data=f_h5['/swire/cdfs/training_indices'])
    indices.create_dataset('validation',
                           data=f_h5['/swire/cdfs/validation_indices'])
    indices.create_dataset('testing',
                           data=f_h5['/swire/cdfs/testing_indices'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', default='crowdastro.h5',
                        help='HDF5 input file')
    parser.add_argument('-o', default='training.h5',
                        help='HDF5 output file')
    args = parser.parse_args()

    with h5py.File(args.i, 'r+') as f_h5:
        with h5py.File(args.o, 'w') as out_f_h5:
            generate(f_h5, out_f_h5)
