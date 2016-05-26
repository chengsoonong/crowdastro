"""Imports and standardises data into crowdastro.

Matthew Alger
The Australian National University
2016
"""

#/usr/bin/env python3

import csv
import logging

import h5py
import numpy

from . import data
from .config import config


def import_atlas(f_h5, f_csv):
    """Imports the ATLAS dataset into crowdastro.

    f_h5: An HDF5 file.
    f_csv: A CSV file.
    """
    with open(config['data_sources']['atlas_catalogue']) as f:
        atlas_catalogue = [l.split() for l in f if not l.startswith('#')]

    # Set up the HDF5 group.
    cdfs = f_h5.create_group('/atlas/cdfs')

    n_skipped = 0  # Number of skipped subjects for debugging.

    # First pass, I'll find coords, names, and Zooniverse IDs, as well as how
    # many data points there are.

    coords = []
    names = []
    zooniverse_ids = []

    for subject in data.get_all_subjects(survey='atlas', field='cdfs'):
        ra, dec = subject['coords']
        zooniverse_id = subject['zooniverse_id']

        # We need the ATLAS name, but we can only get it by going through the
        # ATLAS catalogue and finding the nearest component.
        # https://github.com/chengsoonong/crowdastro/issues/63
        min_dist = float('inf')
        min_obj = None
        for obj in atlas_catalogue:
            # TODO(MatthewJA): Preprocess catalogue to speed this up.
            test_ra = float(obj[4])
            test_dec = float(obj[5])
            dist = numpy.hypot(ra - test_ra, dec - test_dec)
            if dist < min_dist:
                min_dist = dist
                min_obj = obj

        if min_obj is None:
            raise ValueError('No objects in ATLAS catalogue.')

        if min_dist > config['radio_location_threshold']:
            logging.warning('Skipping {}. Nearest ATLAS component is {:.02} '
                            'degrees away.'.format(zooniverse_id, min_dist))
            n_skipped += 1
            continue

        name = min_obj[2]

        # Store the results.
        coords.append((ra, dec))
        names.append(name)
        zooniverse_ids.append(zooniverse_id)

    if n_skipped:
        logging.warning('Skipped {} ATLAS components.'.format(n_skipped))

    n_cdfs = len(names)

    # Sort the data by ATLAS name.
    coords_to_names = dict(zip(coords, names))
    zooniverse_ids_to_names = dict(zip(zooniverse_ids, names))

    coords.sort(key=coords_to_names.get)
    zooniverse_ids.sort(key=zooniverse_ids_to_names.get)
    names.sort()

    # Store coords in HDF5.
    coords = numpy.array(coords)
    coords_ds = cdfs.create_dataset('positions', data=coords)

    # Store Zooniverse IDs and names in CSV.
    writer = csv.writer(f_csv)
    for zooniverse_id, name in zip(zooniverse_ids, names):
        writer.writerow([zooniverse_id, name])

    # Second pass, I'll fetch the images.
    # Allocate space in the HDF5 file.
    dim_2x2 = (n_cdfs, config['sizes']['atlas']['fits_height'],
               config['sizes']['atlas']['fits_width'])
    dim_5x5 = (n_cdfs, config['sizes']['atlas']['fits_height_large'],
               config['sizes']['atlas']['fits_width_large'])
    cdfs_images_2x2 = cdfs.create_dataset('images_2x2', dim_2x2)
    cdfs_images_5x5 = cdfs.create_dataset('images_5x5', dim_5x5)

    for index, zooniverse_id in enumerate(zooniverse_ids):
        subject = data.get_subject(zooniverse_id)
        image_2x2 = data.get_radio(subject, size='2x2')
        cdfs_images_2x2[index] = image_2x2
        image_5x5 = data.get_radio(subject, size='5x5')
        cdfs_images_5x5[index] = image_5x5


if __name__ == '__main__':
    with h5py.File('test.h5', 'w') as f_h5, open('test.csv', 'w') as f_csv:
        import_atlas(f_h5, f_csv)
