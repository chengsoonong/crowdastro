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

def prep_h5(f_h5):
    """Creates hierarchy in HDF5 file."""
    cdfs = f_h5.create_group('/atlas/cdfs')
    swire_cdfs = f_h5.create_group('/swire/cdfs')


def prep_csv(f_csv):
    """Writes headers of CSV."""
    writer = csv.writer(f_csv)
    writer.writerow(['survey', 'field', 'zooniverse_id', 'name'])


def import_atlas(f_h5, f_csv):
    """Imports the ATLAS dataset into crowdastro, as well as associated SWIRE.

    f_h5: An HDF5 file.
    f_csv: A CSV file.
    """
    with open(config['data_sources']['atlas_catalogue']) as f:
        atlas_catalogue = [l.split() for l in f if not l.startswith('#')]

    n_skipped = 0  # Number of skipped subjects for debugging.

    # Fetch groups from HDF5.
    cdfs = f_h5['/atlas/cdfs']
    swire_cdfs = f_h5['/swire/cdfs']

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

        if min_dist > config['surveys']['atlas']['distance_cutoff']:
            logging.warning('Skipping {}. Nearest ATLAS component is {:.02} '
                            'degrees away.'.format(zooniverse_id, min_dist))
            n_skipped += 1
            continue

        name = min_obj[1]

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
        writer.writerow(['atlas', 'cdfs', zooniverse_id, name])

    # Second pass, I'll fetch the images.
    # Allocate space in the HDF5 file.
    dim_2x2 = (n_cdfs, config['surveys']['atlas']['fits_height'],
               config['surveys']['atlas']['fits_width'])
    dim_5x5 = (n_cdfs, config['surveys']['atlas']['fits_height_large'],
               config['surveys']['atlas']['fits_width_large'])
    cdfs_radios_2x2 = cdfs.create_dataset('images_2x2', dim_2x2)
    cdfs_radios_5x5 = cdfs.create_dataset('images_5x5', dim_5x5)
    cdfs_infrareds_2x2 = swire_cdfs.create_dataset('images_2x2', dim_2x2)
    cdfs_infrareds_5x5 = swire_cdfs.create_dataset('images_5x5', dim_5x5)

    for index, zooniverse_id in enumerate(zooniverse_ids):
        subject = data.get_subject(zooniverse_id)
        radio_2x2 = data.get_radio(subject, size='2x2')
        cdfs_radios_2x2[index] = radio_2x2
        radio_5x5 = data.get_radio(subject, size='5x5')
        cdfs_radios_5x5[index] = radio_5x5
        infrared_2x2 = data.get_ir(subject, size='2x2')
        cdfs_infrareds_2x2[index] = infrared_2x2
        infrared_5x5 = data.get_ir(subject, size='5x5')
        cdfs_infrareds_5x5[index] = infrared_5x5


def import_swire(f_h5, f_csv):
    """Imports the SWIRE dataset into crowdastro.

    f_h5: An HDF5 file.
    f_csv: A CSV file.
    """
    names = []
    rows = []
    with open(config['data_sources']['swire_catalogue']) as f_tbl:
        # This isn't a valid ASCII table, so Astropy can't handle it.
        for _ in range(9):  # Skip the first five lines.
            next(f_tbl)

        # Get the column names.
        columns = [c.strip() for c in next(f_tbl).strip().split('|')]

        for _ in range(9):  # Skip the next three lines.
            next(f_tbl)

        for row in f_tbl:
            row = row.strip().split()
            assert len(row) == 156
            row = dict(zip(columns, row))

            name = row['object']
            ra = float(row['ra'])
            dec = float(row['dec'])
            flux_ap2_36 = float(row['flux_ap2_36'])
            flux_ap2_45 = float(row['flux_ap2_45'])
            flux_ap2_58 = float(row['flux_ap2_58'])
            flux_ap2_80 = float(row['flux_ap2_80'])
            flux_ap2_24 = float(row['flux_ap2_24'])
            stell_36 = float(row['stell_36'])
            rows.append((ra, dec, flux_ap2_36, flux_ap2_45, flux_ap2_58,
                         flux_ap2_80, flux_ap2_24, stell_36))
            names.append(name)

    # Sort by name.
    rows_to_names = dict(zip(rows, names))
    rows.sort(key=rows_to_names.get)
    names.sort()

    # Write names to CSV.
    writer = csv.writer(f_csv)
    for name in names:
        writer.writerow(['swire', '', '', name])

    # Write numeric data to HDF5.
    rows = numpy.array(rows)
    f_h5['/swire/cdfs'].create_dataset('catalogue', data=rows)


if __name__ == '__main__':
    with h5py.File('test.h5', 'w') as f_h5, open('test.csv', 'w') as f_csv:
        prep_h5(f_h5)
        prep_csv(f_csv)
        import_atlas(f_h5, f_csv)
        import_swire(f_h5, f_csv)
