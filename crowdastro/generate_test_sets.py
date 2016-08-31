"""Generates sets of testing indices for the galaxy classification task.

Matthew Alger
The Australian National University
2016
"""

import argparse

import h5py
import numpy

from .config import config


ATLAS_WIDTH = config['surveys']['atlas']['fits_width']
ATLAS_HEIGHT = config['surveys']['atlas']['fits_height']
ATLAS_SIZE = ATLAS_WIDTH * ATLAS_HEIGHT


def get_nearby_galaxies(atlas_vector, radius=1/60):
    """Gets all nearby galaxy IDs.

    atlas_vector: crowdastro ATLAS vector.
    radius: Radius to call a galaxy "nearby", in degrees. Default 1 arcmin.
    -> list of ints
    """
    return (atlas_vector[2 + ATLAS_SIZE:] <= radius).nonzero()[0]


def main(f_h5, n, p):
    ir_survey = f_h5.attrs['ir_survey']
    fan_labels = f_h5['/{}/cdfs/fan_labels'.format(ir_survey)].value
    norris_labels = f_h5['/{}/cdfs/norris_labels'.format(ir_survey)].value

    candidates = []
    for atlas_id, atlas in enumerate(f_h5['/atlas/cdfs/numeric']):
        # Get all nearby galaxies.
        nearby = get_nearby_galaxies(atlas)
        # If Norris and Fan agree on all nearby galaxies, then add this ATLAS
        # object to a set of candidates.
        if all(fan_labels[n] == norris_labels[n] for n in nearby):
            candidates.append(atlas_id)

    n_atlas = f_h5['/atlas/cdfs/numeric'].shape[0]
    test_sets = []
    for i in range(n):
        # Select at random, without replacement, candidate ATLAS objects.
        candidates_ = candidates[:]
        numpy.random.shuffle(candidates_)
        atlas_test_set = candidates_[:int(n_atlas * p)]

        # Get all nearby galaxies and add all nearby galaxies to the test set.
        test_set = []
        for atlas_id in atlas_test_set:
            nearby = get_nearby_galaxies(f_h5['/atlas/cdfs/numeric'][atlas_id])
            test_set.extend(nearby)

        test_sets.append(list(set(test_set)))

    # Because the test sets are galaxy-based but the drawing was radio-based,
    # we may have unequal length lists, so we'll crop them.
    min_length = min(len(test_set) for test_set in test_sets)
    test_sets_ = []
    for test_set in test_sets:
        # Have to reshuffle so we don't bias against later IDs.
        numpy.random.shuffle(test_set)
        test_sets_.append(sorted(test_set[:min_length]))

    test_sets = numpy.array(test_sets_)
    f_h5.create_dataset('/{}/cdfs/test_sets'.format(ir_survey),
                        data=test_sets)


def _populate_parser(parser):
    parser.description = 'Generates sets of testing indices for the galaxy ' \
                         'classification task.'
    parser.add_argument('--h5', default='data/crowdastro.h5',
                        help='Crowdastro HDF5 file')
    parser.add_argument('--n', default=5, type=int,
                        help='Number of test sets')
    parser.add_argument('--p', default=0.5, type=float,
                        help='Percentage size of test sets')


def _main(args):
    with h5py.File(args.h5, 'r+') as f_h5:
        main(f_h5, args.n, args.p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
