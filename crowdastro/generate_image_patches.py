"""Generates image patch training data.

Matthew Alger
The Australian National University
2016
"""

import argparse
import csv
import logging
import os.path

import astropy.io.fits
import astropy.wcs
import h5py
import numpy
import sklearn.neighbors

from .config import config


PATCH_RADIUS = config['patch_radius']
ARCMIN = 1 / 60


def get_patch(image, wcs, ra, dec):
    """Gets the image patch associated with x, y for the given object."""
    (x, y), = wcs.all_world2pix([[ra, dec]], 1)
    try:
        if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
            return image[int(x - PATCH_RADIUS) : int(x + PATCH_RADIUS),
                         int(y - PATCH_RADIUS) : int(y + PATCH_RADIUS)]
    except IndexError:
        pass

    return None


def generate(inputs_h5, inputs_csv, training_h5):
    """Generates training image patches.

    inputs_h5: Inputs HDF5 file.
    inputs_csv: Inputs CSV file.
    training_h5: Training HDF5 file.
    """
    if 'raw_patches' in training_h5:
        del training_h5['raw_patches']

    n = len(training_h5['positions'])
    patches = training_h5.create_dataset('raw_patches',
            shape=(n, PATCH_RADIUS * 2, PATCH_RADIUS * 2),
            dtype=float)

    # Each SWIRE object is associated with an ATLAS source it is in the
    # neighbourhood of. This allows finding an image containing that SWIRE
    # object.
    headers = []  # Collect FITS headers.
    logging.debug('Scanning ATLAS data.')
    for row in csv.DictReader(inputs_csv):
        if row['survey'] != 'atlas':
            continue

        assert row['field'] == 'cdfs'
        headers.append(row['header'])

    n_patches = 0  # For debugging.
    images = inputs_h5['/atlas/cdfs/images_5x5']
    atlas_to_image_and_wcs = {}
    logging.debug('Scanning SWIRE data.')
    swire_to_atlas = inputs_h5['/swire/cdfs/catalogue'][:, 8]
    for index, (ra, dec) in enumerate(training_h5['positions']):
        atlas_index = int(swire_to_atlas[index])

        if atlas_index not in atlas_to_image_and_wcs:
            image = images[atlas_index]
            header = headers[atlas_index]
            wcs = astropy.wcs.WCS(astropy.io.fits.Header.fromstring(header))
            atlas_to_image_and_wcs[atlas_index] = (image, wcs)

        image, wcs = atlas_to_image_and_wcs[atlas_index]
        patch = get_patch(image, wcs, ra, dec)
        patches[index] = patch
        n_patches += 1

    logging.debug('Found %d patches.', n_patches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', default='crowdastro.h5',
                        help='HDF5 inputs file')
    parser.add_argument('--csv', default='crowdastro.csv',
                        help='CSV inputs file')
    parser.add_argument('--training', default='training.h5',
                        help='HDF5 training file')
    args = parser.parse_args()

    with h5py.File(args.h5, 'r') as inputs_h5:
        with open(args.csv, 'r') as inputs_csv:
            with h5py.File(args.training, 'r+') as training_h5:
                generate(inputs_h5, inputs_csv, training_h5)
