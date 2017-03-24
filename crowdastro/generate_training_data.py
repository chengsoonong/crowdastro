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
ARCMIN = 1 / 60  # deg
PATCH_SIZE = (config['patch_radius'] * 2) ** 2  # px


def remove_nans(n):
    """Replaces NaN with 0."""
    if numpy.ma.is_masked(n):
        return 0

    return float(n)


def generate_distances(f_h5, ir_prefix, radio_prefix):
    """Generates distance features for infrared objects.

    f_h5: crowdastro HDF5 file.
    ir_prefix: '/survey/field/' for IR data.
    radio_prefix: '/survey/field/' for radio data.
    -> Array of floats.
    """
    # nearby is a boolean array of whether radio/IR objects are nearby.
    # It's too big to load into memory.
    nearby = f_h5[ir_prefix + 'nearby']
    n_ir = nearby.shape[1]
    distances = numpy.zeros((n_ir,))
    has_no_distance = numpy.zeros((n_ir,), dtype=bool)
    for ir_index, ir_info in enumerate(f_h5[ir_prefix + 'numeric']):
        position = ir_info[:2]
        nearby_this = nearby[:, ir_index]
        if not nearby_this.any():
            has_no_distance[ir_index] = 1
            continue

        nearby_positions = f_h5[radio_prefix + 'numeric'][nearby_this, :2]
        distances_this = numpy.linalg.norm(nearby_positions - position, axis=1)
        assert distances_this.shape == nearby_positions.shape[:1]
        distances[ir_index] = distances_this.min()

    # Fill in blanks with the mean value.
    distances[has_no_distance] = distances[~has_no_distance].mean()

    return distances


def generate(f_h5, out_f_h5, radio_survey='atlas', field='cdfs'):
    """Generates potential hosts and their astronomical features.

    f_h5: crowdastro input HDF5 file.
    out_f_h5: Training data output HDF5 file.
    radio_survey: 'first' or 'atlas'.
    field: 'cdfs' or 'elais'; only used for ATLAS.
    """
    FITS_HEIGHT = config['surveys'][radio_survey]['fits_height']
    FITS_WIDTH = config['surveys'][radio_survey]['fits_height']
    IMAGE_SIZE = FITS_HEIGHT * FITS_WIDTH  # px
    ir_survey = f_h5.attrs['ir_survey']

    if radio_survey == 'first' and ir_survey != 'wise':
        raise ValueError('FIRST must use WISE IR data.')

    if radio_survey == 'atlas':
        ir_prefix = '/{}/{}/'.format(ir_survey, field)
        radio_prefix = '/atlas/{}/'.format(field)
    elif radio_survey == 'first':
        ir_prefix = '/wise/first/'
        radio_prefix = '/first/first/'
        field = 'first'

    if ir_survey == 'swire':
        swire = f_h5[ir_prefix + 'numeric']
        fluxes = swire[:, 2:7]
        # Skip stellarities (index 7).
        # Index 8 was used for distances.
        # Indices 9+ are used for images.
        coords = swire[:, :2]
        s1_s2 = fluxes[:, 0] / fluxes[:, 1]
        s2_s3 = fluxes[:, 1] / fluxes[:, 2]
        s3_s4 = fluxes[:, 2] / fluxes[:, 3]
        astro_features = numpy.concatenate(
                [fluxes,
                 s1_s2.reshape((-1, 1)),
                 s2_s3.reshape((-1, 1)),
                 s3_s4.reshape((-1, 1))], axis=1)
    elif ir_survey == 'wise':
        wise = f_h5[ir_prefix + 'numeric']
        magnitudes = wise[:, 2:6]
        # Index 6 was used for distances.
        # Indices 7+ are images.
        coords = wise[:, :2]

        # Magnitude differences are probably useful features.
        w1_w2 = magnitudes[:, 0] - magnitudes[:, 1]
        w2_w3 = magnitudes[:, 1] - magnitudes[:, 2]

        # Converting the magnitudes to a linear scale seems to improve
        # performance.
        linearised_magnitudes = numpy.power(10, -0.4 * magnitudes)
        w1_w2 = numpy.power(10, -0.4 * w1_w2)
        w2_w3 = numpy.power(10, -0.4 * w2_w3)
        astro_features = numpy.concatenate(
                [linearised_magnitudes,
                 w1_w2.reshape((-1, 1)),
                 w2_w3.reshape((-1, 1))], axis=1)

    distances = generate_distances(f_h5, ir_prefix, radio_prefix)

    n_features = config['surveys'][ir_survey]['n_features']
    assert astro_features.shape[1] + 1 == n_features

    # We now need to find the labels for each.
    if field != 'elais':  # No labels for ELAIS-S1.
        truths = set(f_h5[radio_prefix + 'consensus_objects'][:, 1])
        labels = numpy.array([o in truths for o in range(len(astro_features))])

        assert len(labels) == len(astro_features)
        out_f_h5.create_dataset('labels', data=labels)

    # Preallocate space for the features. This is because the image features are
    # very large, and so they may not fit in memory.
    features = out_f_h5.create_dataset('raw_features', dtype='float32',
                                       shape=(astro_features.shape[0],
                                              n_features + PATCH_SIZE))
    out_f_h5.create_dataset('positions', data=coords)
    out_f_h5.attrs['ir_survey'] = ir_survey
    out_f_h5.attrs['field'] = field
    assert len(astro_features) == len(distances)

    # Store the non-image features.
    features[:, :n_features] = numpy.hstack([astro_features, distances.reshape((-1, 1))])
    # Store the image features.
    if ir_survey == 'wise':
        features[:, -PATCH_SIZE:] = wise[:, 7:]
    elif ir_survey == 'swire':
        features[:, -PATCH_SIZE:] = swire[:, 9:]


def _populate_parser(parser):
    parser.description = 'Generates training data (potential hosts and their ' \
                         'astronomical features).'
    parser.add_argument('-i', default='data/crowdastro.h5',
                        help='HDF5 input file')
    parser.add_argument('-o', default='data/training.h5',
                        help='HDF5 output file')
    parser.add_argument('--field', default='cdfs',
                        help='ATLAS field (ATLAS only)')
    parser.add_argument('--survey', default='atlas', choices=['atlas', 'first'],
                        help='Radio survey')


def _main(args):
    with h5py.File(args.i, 'r') as f_h5:
        assert f_h5.attrs['version'] == '0.5.1'
        with h5py.File(args.o, 'w') as out_f_h5:
            generate(f_h5, out_f_h5, field=args.field, radio_survey=args.survey)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
