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


def main(c_h5, t_h5, n, p, p_cnn=0.1, add=False):
    ir_survey = c_h5.attrs['ir_survey']
    ir_survey_ = t_h5.attrs['ir_survey']
    assert ir_survey == ir_survey_
    fan_labels = c_h5['/{}/cdfs/fan_labels'.format(ir_survey)].value
    norris_labels = c_h5['/{}/cdfs/norris_labels'.format(ir_survey)].value

    if '/{}/cdfs/test_sets'.format(ir_survey) in c_h5 and not add:
        raise ValueError('Test sets already exist.')

    candidates = []
    for atlas_id, atlas in enumerate(c_h5['/atlas/cdfs/numeric']):
        # Get all nearby galaxies.
        nearby = get_nearby_galaxies(atlas)
        # If Norris and Fan agree on all nearby galaxies, then add this ATLAS
        # object to a set of candidates.
        if all(fan_labels[n] == norris_labels[n] for n in nearby):
            candidates.append(atlas_id)

    n_atlas = c_h5['/atlas/cdfs/numeric'].shape[0]

    if not add:
        # Pull out p_cnn% of the ATLAS objects for use training the CNN.
        cnn_train_set_atlas = list(range(n_atlas))
        numpy.random.shuffle(cnn_train_set_atlas)
        cnn_train_set_atlas = set(cnn_train_set_atlas[:int(n_atlas * p_cnn)])
        cnn_train_set = []
        for atlas_id in cnn_train_set_atlas:
            nearby = get_nearby_galaxies(c_h5['/atlas/cdfs/numeric'][atlas_id])
            cnn_train_set.extend(nearby)
        cnn_train_set.sort()
        cnn_train_set_ = set(cnn_train_set)
    else:
        cnn_train_set = set(
            c_h5['/{}/cdfs/cnn_train_set'.format(ir_survey)].value)
        cnn_train_set_atlas = set()
        for atlas_id, atlas in enumerate(c_h5['/atlas/cdfs/numeric']):
            nearby = get_nearby_galaxies(atlas)
            for i in nearby:
                if i in cnn_train_set:
                    cnn_train_set_atlas.add(atlas_id)
                    break
        cnn_train_set = sorted(cnn_train_set)
        cnn_train_set_ = set(cnn_train_set)

    # Generate the test sets. Make sure none of the ATLAS objects are in the
    # CNN training set.
    test_sets = []
    if add:
        for test_set in c_h5['/{}/cdfs/test_sets'.format(ir_survey)].value:
            test_sets.append(list(test_set))
        assert len(test_sets) == c_h5['/{}/cdfs/test_sets'.format(ir_survey)
                                      ].shape[0]
    for i in range(n):
        # Select at random, without replacement, candidate ATLAS objects.
        candidates_ = list(set(candidates) - cnn_train_set_atlas)
        numpy.random.shuffle(candidates_)
        atlas_test_set = []
        while len(atlas_test_set) < int(n_atlas * p):
            for atlas_id in candidates_:
                assert atlas_id not in cnn_train_set_atlas
                nearby = get_nearby_galaxies(
                    c_h5['/atlas/cdfs/numeric'][atlas_id])
                for j in nearby:
                    # Check for overlaps.
                    if j in cnn_train_set_:
                        break
                else:
                    # No overlaps!
                    atlas_test_set.append(atlas_id)

        # Get all nearby galaxies and add all nearby galaxies to the test set.
        test_set = []
        for atlas_id in atlas_test_set:
            assert atlas_id not in cnn_train_set_atlas
            nearby = get_nearby_galaxies(c_h5['/atlas/cdfs/numeric'][atlas_id])
            for j in nearby:
                # Have to make sure there's no overlaps here.
                assert j not in cnn_train_set_
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
    if add:
        del c_h5['/{}/cdfs/test_sets'.format(ir_survey)]
        del t_h5['test_sets']
    c_h5.create_dataset('/{}/cdfs/test_sets'.format(ir_survey),
                        data=test_sets)
    t_h5.create_dataset('test_sets', data=test_sets)

    if not add:
        cnn_train_set = numpy.array(cnn_train_set)
        c_h5.create_dataset('/{}/cdfs/cnn_train_set'.format(ir_survey),
                            data=cnn_train_set)
        t_h5.create_dataset('cnn_train_set', data=cnn_train_set)
    else:
        assert 'cnn_train_set' in c_h5['/{}/cdfs/'.format(ir_survey)]
        assert 'cnn_train_set' in t_h5


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
