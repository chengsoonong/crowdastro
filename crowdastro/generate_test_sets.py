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


def main(c_h5, t_h5, n, p, add=False, field='cdfs'):
    ir_survey = c_h5.attrs['ir_survey']
    ir_survey_ = t_h5.attrs['ir_survey']
    assert ir_survey == ir_survey_

    if '/{}/{}/test_sets'.format(ir_survey, field) in c_h5 and not add:
        raise ValueError('Test sets already exist.')

    candidates = []
    for atlas_id, atlas in enumerate(c_h5['/atlas/{}/numeric'.format(field)]):
        candidates.append(atlas_id)

    n_atlas = c_h5['/atlas/{}/numeric'.format(field)].shape[0]

    # Generate the test sets.
    test_sets = []
    if add:
        for test_set in c_h5['/{}/{}/test_sets'.format(ir_survey, field)].value:
            test_sets.append(list(test_set))
        assert len(test_sets) == c_h5['/{}/{}/test_sets'.format(
                ir_survey, field)].shape[0]
    for i in range(n):
        # Select at random, without replacement, candidate ATLAS objects.
        candidates_ = list(set(candidates))
        numpy.random.shuffle(candidates_)
        atlas_test_set = []
        for atlas_id in candidates_:
            if len(atlas_test_set) >= int(n_atlas * p):
                break

            atlas_test_set.append(atlas_id)

        # Get all nearby galaxies and add all nearby galaxies to the test set.
        test_set = []
        for atlas_id in atlas_test_set:
            nearby = get_nearby_galaxies(
                c_h5['/atlas/{}/numeric'.format(field)][atlas_id])
            test_set.extend(nearby)

        test_set = sorted(set(test_set))
        if test_sets:
            assert test_sets[-1] != test_set[:len(test_sets[-1])]
        test_sets.append(test_set)

    # Because the test sets are galaxy-based but the drawing was radio-based,
    # we may have unequal length lists, so we'll crop them.
    min_length = min(len(test_set) for test_set in test_sets)
    test_sets_ = []
    for test_set in test_sets:
        # Have to reshuffle so we don't bias against later IDs.
        numpy.random.shuffle(test_set)
        test_sets_.append(sorted(test_set[:min_length]))

    test_sets = numpy.array(test_sets_)
    if add:
        del c_h5['/{}/{}/test_sets'.format(ir_survey, field)]
        del t_h5['test_sets']
    c_h5.create_dataset('/{}/{}/test_sets'.format(ir_survey, field),
                        data=test_sets)
    t_h5.create_dataset('test_sets', data=test_sets)


def _populate_parser(parser):
    parser.description = 'Generates sets of testing indices for the galaxy ' \
                         'classification task.'
    parser.add_argument('--crowdastro', default='data/crowdastro.h5',
                        help='Crowdastro HDF5 file')
    parser.add_argument('--training', default='data/training.h5',
                        help='Training HDF5 file')
    parser.add_argument('--n', default=5, type=int,
                        help='Number of test sets')
    parser.add_argument('--p', default=0.5, type=float,
                        help='Percentage size of test sets')
    parser.add_argument('--field', default='cdfs',
                        help='ATLAS field', choices=('cdfs', 'elais'))
    parser.add_argument('--add', action='store_true',
                        help='Add new test sets to existing test sets')


def _main(args):
    with h5py.File(args.crowdastro, 'r+') as c_h5:
        with h5py.File(args.training, 'r+') as t_h5:
            main(c_h5, t_h5, args.n, args.p, add=args.add)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
