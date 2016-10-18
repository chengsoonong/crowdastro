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


def generate(f_h5, out_f_h5, field='cdfs'):
    """Generates potential hosts and their astronomical features.

    f_h5: crowdastro input HDF5 file.
    out_f_h5: Training data output HDF5 file.
    """
    ir_survey = f_h5.attrs['ir_survey']
    if ir_survey == 'swire':
        swire = f_h5['/swire/{}/numeric'.format(field)]
        fluxes = swire[:, 2:7]
        # Skip stellarities.
        distances = swire[:, 8].reshape((-1, 1))
        images = swire[:, 9:]
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
        wise = f_h5['/wise/{}/numeric'.format(field)]
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
        astro_features = numpy.concatenate(
                [linearised_magnitudes,
                 w1_w2.reshape((-1, 1)),
                 w2_w3.reshape((-1, 1))], axis=1)

    n_features = config['surveys'][ir_survey]['n_features']
    assert astro_features.shape[1] + distances.shape[1] == n_features

    # We now need to find the labels for each.
    if field == 'cdfs':
      truths = set(f_h5['/atlas/cdfs/consensus_objects'][:, 1])
      labels = numpy.array([o in truths for o in range(len(astro_features))])

      assert len(labels) == len(astro_features)
      out_f_h5.create_dataset('labels', data=labels)

    assert len(astro_features) == len(distances)
    assert len(distances) == len(images)

    features = numpy.hstack([astro_features, distances, images])
    n_astro = features.shape[1] - images.shape[1]

    # Save to HDF5.
    out_f_h5.create_dataset('raw_features', data=features)
    out_f_h5.create_dataset('positions', data=coords)
    out_f_h5.attrs['ir_survey'] = ir_survey
    out_f_h5.attrs['field'] = field

def _populate_parser(parser):
    parser.description = 'Generates training data (potential hosts and their ' \
                         'astronomical features).'
    parser.add_argument('-i', default='data/crowdastro.h5',
                        help='HDF5 input file')
    parser.add_argument('-o', default='data/training.h5',
                        help='HDF5 output file')
    parser.add_argument('--field', default='cdfs',
                        help='ATLAS field')

def _main(args):
    with h5py.File(args.i, 'r') as f_h5:
        assert f_h5.attrs['version'] == '0.5.1'
        with h5py.File(args.o, 'w') as out_f_h5:
            generate(f_h5, out_f_h5,field=args.field)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
