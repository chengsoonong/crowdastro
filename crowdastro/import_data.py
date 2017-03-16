"""Imports and standardises data into crowdastro.

Matthew Alger
The Australian National University
2016
"""
import argparse
import csv
import hashlib
import logging
import os
import re
import time

from astropy.coordinates import SkyCoord
import astropy.io.fits
from astropy.io import ascii
import astropy.utils.exceptions
import astropy.wcs
import h5py
import numpy
import scipy.spatial
import scipy.spatial.distance
import sklearn.neighbors

from .config import config
from .exceptions import CatalogueError

VERSION = '0.5.1'  # Data version, not module version!
# max number of components * individual component signature size.
MAX_RADIO_SIGNATURE_LENGTH = 50
MAX_NAME_LENGTH = 50  # b
MAX_ZOONIVERSE_ID_LENGTH = 20  # b
PATCH_RADIUS = config['patch_radius']  # px
ARCMIN = 1 / 60  # deg
CANDIDATE_RADIUS = ARCMIN  # deg
FITS_CONVENTION = 1


def hash_file(f):
    """Finds the MD5 hash of a file.

    File must be opened in bytes mode.
    """
    h = hashlib.md5()
    chunk_size = 65536  # 64 KiB
    for chunk in iter(lambda: f.read(chunk_size), b''):
        h.update(chunk)
    return h.hexdigest()


def checksum_file(filename, h):
    """Checks files hash to expected hashes.

    filename: str.
    h: Hex hash string to compare against.

    -> True iff file matches hash.
    """
    with open(filename, 'rb') as f:
        h_ = hash_file(f)
        return h_ == h


def prep_h5(f_h5, ir_survey):
    """Creates hierarchy in HDF5 file."""
    f_h5.create_group('/first/first')
    f_h5.create_group('/atlas/cdfs')
    f_h5.create_group('/atlas/elais')
    f_h5.create_group('/{}/cdfs'.format(ir_survey))
    f_h5.create_group('/{}/elais'.format(ir_survey))
    f_h5.create_group('/{}/first'.format(ir_survey))
    f_h5.attrs['version'] = VERSION
    f_h5.attrs['ir_survey'] = ir_survey


def import_atlas(f_h5, test=False, field='cdfs'):
    """Imports the ATLAS dataset into crowdastro, as well as associated SWIRE.

    f_h5: An HDF5 file.
    test: Flag to run on only 10 subjects. Default False.
    """
    from . import rgz_data as data

    # Fetch groups from HDF5.
    cdfs = f_h5['/atlas/{}'.format(field)]

    # First pass, I'll find coords, names, and Zooniverse IDs, as well as how
    # many data points there are.

    coords = []
    names = []
    zooniverse_ids = []

    if (field == 'cdfs'):
        # We need the ATLAS name, but we can only get it by going through the
        # ATLAS catalogue and finding the nearest component.
        # https://github.com/chengsoonong/crowdastro/issues/63
        # Fortunately, @jbanfield has already done this, so we can just load
        # that CSV and match the names.
        # TODO(MatthewJA): This matches the ATLAS component ID, but maybe we
        # should be using the name instead.
        rgz_to_atlas = {}
        with open(config['data_sources']['rgz_to_atlas']) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rgz_to_atlas[row['ID_RGZ']] = row['ID']

        all_subjects = data.get_all_subjects(survey='atlas', field=field)
        if test:
            all_subjects = all_subjects.limit(10)

        for subject in all_subjects:
            ra, dec = subject['coords']
            zooniverse_id = subject['zooniverse_id']

            rgz_source_id = subject['metadata']['source']
            if rgz_source_id not in rgz_to_atlas:
                logging.debug('Skipping %s; no matching ATLAS component.',
                              zooniverse_id)
                continue
            name = rgz_to_atlas[rgz_source_id]

            # Store the results.
            coords.append((ra, dec))
            names.append(name)
            zooniverse_ids.append(zooniverse_id)

    elif (field == 'elais'):
        atlascatalogue = ascii.read(config['data_sources']['atlas_catalogue'])
        ras, decs = atlascatalogue['RA_deg'], atlascatalogue['Dec_deg']
        e_ids = atlascatalogue['ID']
        fields = atlascatalogue['field']

        # Store the results.
        for ra, dec, e_id, field_ in zip(ras, decs, e_ids, fields):
            if (field_ == 'ELAIS-S1'):
                coords.append((ra, dec))
                names.append(e_id)
                zooniverse_ids.append(e_id)

    n_cdfs = len(names)

    # Sort the data by Zooniverse ID.
    coords_to_zooniverse_ids = dict(zip(coords, zooniverse_ids))
    names_to_zooniverse_ids = dict(zip(names, zooniverse_ids))

    coords.sort(key=coords_to_zooniverse_ids.get)
    names.sort(key=names_to_zooniverse_ids.get)
    zooniverse_ids.sort()

    # Begin to store the data. We will have two tables: one for numeric data,
    # and one for strings. We will have to preallocate the numeric table so that
    # we aren't storing huge amounts of image data in memory.

    # Strings.
    dtype = [('zooniverse_id', '<S{}'.format(MAX_ZOONIVERSE_ID_LENGTH)),
             ('name', '<S{}'.format(MAX_NAME_LENGTH))]
    string_data = numpy.array(list(zip(zooniverse_ids, names)), dtype=dtype)
    cdfs.create_dataset('string', data=string_data, dtype=dtype)

    # Numeric.
    image_size = (config['surveys']['atlas']['fits_width'] *
                  config['surveys']['atlas']['fits_height'])
    # RA, DEC, radio
    dim = (n_cdfs, 1 + 1 + image_size)
    numeric = cdfs.create_dataset('numeric', shape=dim, dtype='float32')

    # Load image patches and store numeric data.
    with astropy.io.fits.open(
            config['data_sources']['atlas_{}_image'.format(field)],
            ignore_blank=True) as atlas_image:
        wcs = astropy.wcs.WCS(atlas_image[0].header).dropaxis(3).dropaxis(2)
        pix_coords = wcs.all_world2pix(coords, FITS_CONVENTION)
        assert pix_coords.shape[1] == 2
        logging.debug('Fetching %d ATLAS images.', len(pix_coords))
        for index, (x, y) in enumerate(pix_coords):
            radio = atlas_image[0].data[
                0, 0,  # stokes, freq
                int(y) - config['surveys']['atlas']['fits_height'] // 2:
                int(y) + config['surveys']['atlas']['fits_height'] // 2,
                int(x) - config['surveys']['atlas']['fits_width'] // 2:
                int(x) + config['surveys']['atlas']['fits_width'] // 2]
            numeric[index, 0] = coords[index][0]
            numeric[index, 1] = coords[index][1]
            numeric[index, 2:2 + image_size] = radio.reshape(-1)

    logging.debug('ATLAS imported.')


def import_first(f_h5, test=False):
    """Imports the FIRST dataset into crowdastro.

    f_h5: An HDF5 file.
    test: Flag to run on only 10 subjects. Default False.
    """
    from . import rgz_data as data

    # Fetch groups from HDF5.
    first = f_h5['/first/first']

    # We first need:
    # - coords,
    # - names, and
    # - Zooniverse IDs.
    # This can all be dumped from the Radio Galaxy Zoo database.

    coords = []
    names = []
    zooniverse_ids = []

    first_objects = data.db.radio_subjects.find({'metadata.survey': 'first'})
    if test:
        first_objects = first_objects.limit(10)

    for first_object in first_objects:
        ra, dec = first_object['coords']
        name = first_object['metadata']['source']
        zooniverse_id = first_object['zooniverse_id']

        coords.append((ra, dec))
        names.append(name)
        zooniverse_ids.append(zooniverse_id)

    n_first = len(names)

    # Sort the data by Zooniverse ID.
    coords_to_zooniverse_ids = dict(zip(coords, zooniverse_ids))
    names_to_zooniverse_ids = dict(zip(names, zooniverse_ids))

    coords.sort(key=coords_to_zooniverse_ids.get)
    names.sort(key=names_to_zooniverse_ids.get)
    zooniverse_ids.sort()

    # Begin to store the data. We will have two tables: one for numeric data,
    # and one for strings. We will have to preallocate the numeric table so that
    # we aren't storing huge amounts of image data in memory.

    # Strings.
    dtype = [('zooniverse_id', '<S{}'.format(MAX_ZOONIVERSE_ID_LENGTH)),
             ('name', '<S{}'.format(MAX_NAME_LENGTH))]
    string_data = numpy.array(list(zip(zooniverse_ids, names)), dtype=dtype)
    first.create_dataset('string', data=string_data, dtype=dtype)

    # Numeric.
    # RA, DEC
    dim = (n_first, 2)
    numeric = first.create_dataset('numeric', shape=dim, dtype='float32')
    numeric[:, :] = numpy.array(coords)

    # With ATLAS, we would now load image patches etc. However, this leads to a
    # *massive* file for FIRST, so we will not do that and instead just fetch
    # the images from the filesystem when necessary. Note that RGZ stores the
    # FIRST images on a one subject = one file basis.

    logging.debug('FIRST imported.')


def remove_nulls(n):
    """Swaps nulls with zeros."""
    if n == 'null':
        return 0

    return n


def import_swire(f_h5, field='cdfs'):
    """Imports the SWIRE dataset into crowdastro.

    f_h5: An HDF5 file.
    field: 'cdfs' or 'elais'.
    """
    names = []
    rows = []
    logging.debug('Reading SWIRE catalogue.')
    with open(
        config['data_sources']['swire_{}_catalogue'.format(field)]
    ) as f_tbl:
        # This isn't a valid ASCII table, so Astropy can't handle it. This means
        # we have to parse it manually.
        if field == 'cdfs':
            for _ in range(5):  # Skip the first five lines.
                next(f_tbl)

            # Get the column names.
            columns = [c.strip() for c in next(f_tbl).strip().split('|')][1:-1]
            assert len(columns) == 156

            for _ in range(3):  # Skip the next three lines.
                next(f_tbl)

            for row in f_tbl:
                row = row.strip().split()
                assert len(row) == 156
                row = dict(zip(columns, row))
                name = row['object']
                ra = float(row['ra'])
                dec = float(row['dec'])
                flux_ap2_36 = float(remove_nulls(row['flux_ap2_36']))
                flux_ap2_45 = float(remove_nulls(row['flux_ap2_45']))
                flux_ap2_58 = float(remove_nulls(row['flux_ap2_58']))
                flux_ap2_80 = float(remove_nulls(row['flux_ap2_80']))
                flux_ap2_24 = float(remove_nulls(row['flux_ap2_24']))
                stell_36 = float(remove_nulls(row['stell_36']))
                # Extra -1 is so we can store nearest distance later.
                rows.append((ra, dec, flux_ap2_36, flux_ap2_45, flux_ap2_58,
                             flux_ap2_80, flux_ap2_24, stell_36, -1))
                names.append(name)
        elif field == 'elais':
            for _ in range(121):  # Skip the first 121 lines.
                next(f_tbl)

            # Get the column names.
            columns = [c.strip() for c in next(f_tbl).strip().split('|')][1:-1]
            assert len(columns) == 54

            for _ in range(3):  # Skip the next three lines.
                next(f_tbl)

            for row in f_tbl:
                row = row.strip().split()
                assert len(row) == 54
                row = dict(zip(columns, row))
                name = row['object']
                ra = float(row['ra'])
                dec = float(row['dec'])
                flux_ap2_36 = float(remove_nulls(row['flux_ap2_36']))
                flux_ap2_45 = float(remove_nulls(row['flux_ap2_45']))
                flux_ap2_58 = float(remove_nulls(row['flux_ap2_58']))
                flux_ap2_80 = float(remove_nulls(row['flux_ap2_80']))
                flux_ap2_24 = float(remove_nulls(row['flux_ap2_24']))
                stell_36 = float(remove_nulls(row['stell_36']))
                # Extra -1 is so we can store nearest distance later.
                rows.append((ra, dec, flux_ap2_36, flux_ap2_45, flux_ap2_58,
                             flux_ap2_80, flux_ap2_24, stell_36, -1))
                names.append(name)

    logging.debug('Found %d SWIRE objects.', len(names))

    # Sort by name.
    rows_to_names = dict(zip(rows, names))
    rows.sort(key=rows_to_names.get)
    names.sort()

    names = numpy.array(names, dtype='<S{}'.format(MAX_NAME_LENGTH))
    rows = numpy.array(rows)

    # Filter on distance - only include image data for SWIRE objects within a
    # given radius of an ATLAS object. Otherwise, there's way too much data to
    # store.
    swire_positions = rows[:, :2]
    atlas_positions = f_h5['/atlas/{}/numeric'.format(field)][:, :2]
    logging.debug('Computing SWIRE k-d tree.')
    swire_tree = sklearn.neighbors.KDTree(swire_positions, metric='euclidean')
    swire_near_atlas = swire_tree.query_radius(
        atlas_positions, CANDIDATE_RADIUS)
    indices = numpy.concatenate(swire_near_atlas)
    indices = numpy.unique(indices)

    logging.debug('Found %d SWIRE objects near ATLAS objects.', len(indices))

    names = names[indices]
    rows = rows[indices]
    swire_positions = swire_positions[indices]

    # Get distances.
    logging.debug('Finding ATLAS-SWIRE object distances.')
    distances = numpy.zeros((len(atlas_positions), len(swire_positions)),
                            dtype=bool)
    for atlas_index, swire_indices in enumerate(swire_near_atlas):
        distances[atlas_index, swire_indices] = True
    assert distances.shape[0] == atlas_positions.shape[0]
    assert distances.shape[1] == swire_positions.shape[0]
    logging.debug('Done finding distances.')

    # Write numeric data to HDF5.
    f_h5['/swire/{}/'.format(field)].create_dataset(
        'nearby',
        data=distances,
        dtype=bool)

    image_size = (PATCH_RADIUS * 2) ** 2
    dim = (rows.shape[0], rows.shape[1] + image_size)
    numeric = f_h5['/swire/{}'.format(field)].create_dataset(
        'numeric', shape=dim, dtype='float32')
    numeric[:, :rows.shape[1]] = rows
    f_h5['/swire/{}'.format(field)].create_dataset('string', data=names)

    # Load and store radio images.
    logging.debug('Importing radio patches.')
    with astropy.io.fits.open(
            config['data_sources']['atlas_{}_image'.format(field)],
            ignore_blank=True) as atlas_image:
        wcs = astropy.wcs.WCS(atlas_image[0].header).dropaxis(3).dropaxis(2)
        pix_coords = wcs.all_world2pix(swire_positions, FITS_CONVENTION)
        assert pix_coords.shape[1] == 2
        assert pix_coords.shape[0] == len(indices)
        logging.debug('Fetching %d ATLAS patches.', len(indices))

        for index, (x, y) in enumerate(pix_coords):
            radio = atlas_image[0].data[
                0, 0,  # stokes, freq
                int(y) - PATCH_RADIUS:
                int(y) + PATCH_RADIUS,
                int(x) - PATCH_RADIUS:
                int(x) + PATCH_RADIUS]
            numeric[index, -image_size:] = radio.reshape(-1)


def import_wise(f_h5, radio_survey='atlas', field='cdfs'):
    """Imports the WISE dataset into crowdastro.

    f_h5: An HDF5 file.
    radio_survey: 'atlas' or 'first'.
    field: 'cdfs' or 'elais' (only if radio_survey == 'atlas').
    """
    if radio_survey == 'atlas':
        radio_prefix = '/atlas/' + field + '/'
        ir_prefix = '/wise/' + field + '/'
        wise_path = config['data_sources']['wise_{}_catalogue'.format(field)]
    elif radio_survey == 'first':
        radio_prefix = '/first/first/'
        ir_prefix = '/wise/first/'
        wise_path = config['data_sources']['wise_first_catalogue']

    names = []
    rows = []
    logging.debug('Reading WISE catalogue.')
    with open(wise_path) as f_tbl:
        # This isn't a valid ASCII table, so Astropy can't handle it. This means
        # we have to parse it manually.
        if radio_survey == 'atlas':
            n_header_rows = 105
            # Get the column names.
            for _ in range(n_header_rows):  # Skip the first header lines.
                next(f_tbl)
            columns = [c.strip() for c in next(f_tbl).strip().split('|')][1:-1]
            assert len(columns) == 45
            for _ in range(3):  # Skip the next three lines.
                next(f_tbl)

        elif radio_survey == 'first':
            n_header_rows = 34  # Up to but not including header row.
            for _ in range(n_header_rows):  # Skip the first header lines.
                next(f_tbl)
            columns = [c.strip() for c in next(f_tbl).strip().split('|')][1:-1]
            assert len(columns) == 12
            for _ in range(3):  # Skip the next three lines.
                next(f_tbl)

        col_count = len(columns)

        for row in f_tbl:
            row = row.strip().split()
            assert len(row) == col_count
            row = dict(zip(columns, row))
            name = row['designation']
            ra = float(row['ra'])
            dec = float(row['dec'])
            w1mpro = float(remove_nulls(row['w1mpro']))
            w2mpro = float(remove_nulls(row['w2mpro']))
            w3mpro = float(remove_nulls(row['w3mpro']))
            w4mpro = float(remove_nulls(row['w4mpro']))
            # Extra -1 is so we can store nearest distance later.
            rows.append((ra, dec, w1mpro, w2mpro, w3mpro, w4mpro, -1))
            names.append(name)

    n_wise = len(names)
    logging.debug('Found %d WISE objects.', n_wise)

    # Sort by name.
    rows_to_names = dict(zip(rows, names))
    rows.sort(key=rows_to_names.get)
    names.sort()

    names = numpy.array(names, dtype='<S{}'.format(MAX_NAME_LENGTH))
    rows = numpy.array(rows)

    # Filter on distance - only include image data for WISE objects within a
    # given radius of a radio object. Otherwise, there's way too much data to
    # store.
    wise_positions = rows[:, :2]
    radio_positions = f_h5[radio_prefix + 'numeric'][:, :2]
    logging.debug('Computing WISE k-d tree.')
    t = time.time()  # To time KDTree generation.
    # balanced_tree = False switches to the midpoint rule which works better for
    # large datasets like this one.
    wise_tree = scipy.spatial.cKDTree(wise_positions, balanced_tree=False)
    logging.debug('Computing WISE k-d tree took {:.02f} seconds.'.format(
        time.time() - t))
    wise_near_radio = wise_tree.query_ball_point(
        radio_positions, CANDIDATE_RADIUS)
    indices = numpy.concatenate(wise_near_radio)
    indices = numpy.unique(indices).astype('int')

    logging.debug('Found %d WISE objects near radio objects.', len(indices))
    assert sorted(indices) == list(indices)

    names = names[indices]
    rows = rows[indices]
    wise_positions = wise_positions[indices]

    wise_tree = scipy.spatial.cKDTree(wise_positions, balanced_tree=False)
    wise_near_radio = wise_tree.query_ball_point(
        radio_positions, CANDIDATE_RADIUS)

    # Get distances.
    distances = f_h5[ir_prefix].create_dataset(
        'nearby',
        shape=(len(radio_positions), len(wise_positions)),
        dtype=bool)
    logging.debug('Finding radio object-WISE object distances.')
    for radio_index, wise_indices in enumerate(wise_near_radio):
        distances_ = distances[radio_index, :]
        distances_[sorted(wise_indices)] = True
    assert distances.shape[0] == radio_positions.shape[0]
    assert distances.shape[1] == wise_positions.shape[0]
    logging.debug('Done finding distances.')

    image_size = (PATCH_RADIUS * 2) ** 2
    dim = (rows.shape[0], rows.shape[1] + image_size)
    numeric = f_h5[ir_prefix].create_dataset(
        'numeric', shape=dim, dtype='float32')
    numeric[:, :rows.shape[1]] = rows
    f_h5[ir_prefix].create_dataset('string', data=names)

    # Load and store radio images.
    if radio_survey == 'atlas':
        logging.debug('Importing ATLAS radio patches.')
        with astropy.io.fits.open(
                config['data_sources']['atlas_{}_image'.format(field)],
                ignore_blank=True) as atlas_image:
            wcs = astropy.wcs.WCS(atlas_image[0].header).dropaxis(3).dropaxis(2)
            pix_coords = wcs.all_world2pix(wise_positions, FITS_CONVENTION)
            assert pix_coords.shape[1] == 2
            assert pix_coords.shape[0] == len(indices)
            logging.debug('Fetching %d ATLAS patches.', len(indices))

            for index, (x, y) in enumerate(pix_coords):
                radio = atlas_image[0].data[
                    0, 0,  # stokes, freq
                    int(y) - PATCH_RADIUS:
                    int(y) + PATCH_RADIUS,
                    int(x) - PATCH_RADIUS:
                    int(x) + PATCH_RADIUS]
                numeric[index, -image_size:] = radio.reshape(-1)
    elif radio_survey == 'first':
        # Since there isn't just one big image for FIRST, unlike ATLAS, we need
        # to load each individual file.
        # For each FIRST image...
        for dirpath, dirnames, filenames in os.walk(
                config['data_sources']['first_images_dir']):
            for filename in filenames:
                if not filename.endswith('.fits'):
                    continue

                logging.debug('Reading FIRST image: {}'.format(filename))

                image = astropy.io.fits.open(os.path.join(dirpath, filename))
                wcs = astropy.wcs.WCS(image[0].header)
                wcs = wcs.dropaxis(3).dropaxis(2)  # Drop STOKES and FREQ axes.
                min_pix_x = min_pix_y = 0
                _, _, max_pix_y, max_pix_x = image[0].data.shape
                pixel_coords = wcs.all_world2pix(wise_positions, 1)
                valid = numpy.logical_and(
                    numpy.logical_and(min_pix_x <= pixel_coords[:, 0],
                                      pixel_coords[:, 0] < max_pix_x),
                    numpy.logical_and(min_pix_y <= pixel_coords[:, 1],
                                      pixel_coords[:, 1] < max_pix_y)
                    ).nonzero()[0]
                for wise_index, (x, y) in zip(valid, pixel_coords[valid]):
                    x = int(x)
                    y = int(y)
                    patch = image[0].data[
                        0, 0, y - PATCH_RADIUS:y + PATCH_RADIUS,
                        x - PATCH_RADIUS:x + PATCH_RADIUS]
                    if patch.shape[0] == 0 or patch.shape[1] == 0:
                        continue
                    if patch.shape != (PATCH_RADIUS * 2, PATCH_RADIUS * 2):
                        pad_width = numpy.array(
                            [PATCH_RADIUS * 2, PATCH_RADIUS * 2]
                            ) - numpy.array(patch.shape)
                        pad_width = numpy.stack(
                            [numpy.zeros(2), pad_width]).T.astype(int)
                        patch = numpy.pad(patch, pad_width, mode='edge')
                    if numpy.isnan(patch).all():
                        continue

                    numeric[wise_index, -image_size:] = \
                        numpy.nan_to_num(patch).ravel()


def import_norris(f_h5):
    """Imports the Norris et al. (2006) labels.

    f_h5: crowdastro HDF5 file with WISE or SWIRE already imported.
    """
    ir_survey = f_h5.attrs['ir_survey']
    ir_positions = f_h5['/{}/cdfs/numeric'.format(ir_survey)][:, :2]
    ir_tree = sklearn.neighbors.KDTree(ir_positions)
    norris_dat = astropy.io.ascii.read(config['data_sources']['norris_coords'])
    norris_swire = norris_dat['SWIRE']
    norris_coords = []
    for s in norris_swire:
        s = s.strip()
        if len(s) < 19:
            continue

        # e.g. J032931.44-281722.0
        ra_hr = s[1:3]
        ra_min = s[3:5]
        ra_sec = s[5:10]
        dec_sgn = s[10]
        dec_deg = s[11:13]
        dec_min = s[13:15]
        dec_sec = s[15:19]

        ra = '{} {} {}'.format(ra_hr, ra_min, ra_sec)
        dec = '{}{} {} {}'.format(dec_sgn, dec_deg, dec_min, dec_sec)
        logging.debug('Reading Norris coordinate: {}; {}'.format(ra, dec))
        coord = SkyCoord(ra=ra, dec=dec,
                         unit=('hourangle, deg'))
        norris_coords.append(coord)

    norris_labels = numpy.zeros((ir_positions.shape[0],))
    for skycoord in norris_coords:
        # Find a neighbour.
        ra = skycoord.ra.degree
        dec = skycoord.dec.degree
        ((dist,),), ((ir,),) = ir_tree.query([(ra, dec)])
        if dist < config['surveys'][ir_survey]['distance_cutoff']:
            norris_labels[ir] = 1
    f_h5.create_dataset('/{}/cdfs/norris_labels'.format(ir_survey),
                        data=norris_labels)


def import_fan(f_h5):
    """Imports the Fan et al. (2015) labels.

    f_h5: crowdastro HDF5 file with WISE or SWIRE already imported.
    """
    ir_survey = f_h5.attrs['ir_survey']
    ir_names = f_h5['/{}/cdfs/string'.format(ir_survey)]
    ir_positions = f_h5['/{}/cdfs/numeric'.format(ir_survey)][:, :2]
    ir_tree = sklearn.neighbors.KDTree(ir_positions)
    fan_coords = []
    with open(config['data_sources']['fan_swire'], 'r') as fan_dat:
        for row in csv.DictReader(fan_dat):
            ra_hr = row['swire'][8:10]
            ra_min = row['swire'][10:12]
            ra_sec = row['swire'][12:17]
            dec_sgn = row['swire'][17]
            dec_deg = row['swire'][18:20]
            dec_min = row['swire'][20:22]
            dec_sec = row['swire'][22:26]

            ra = '{} {} {}'.format(ra_hr, ra_min, ra_sec)
            dec = '{}{} {} {}'.format(dec_sgn, dec_deg, dec_min, dec_sec)
            fan_coords.append((ra, dec))

    fan_labels = numpy.zeros((ir_positions.shape[0],))
    for ra, dec in fan_coords:
        # Find a neighbour.
        skycoord = SkyCoord(ra=ra, dec=dec, unit=('hourangle', 'deg'))
        ra = skycoord.ra.degree
        dec = skycoord.dec.degree
        ((dist,),), ((ir,),) = ir_tree.query([(ra, dec)])
        if dist < config['surveys'][ir_survey]['distance_cutoff']:
            fan_labels[ir] = 1
    f_h5.create_dataset('/{}/cdfs/fan_labels'.format(ir_survey),
                         data=fan_labels)


def contains(bbox, point):
    """Checks if point is within bbox.

    bbox: [[x0, x1], [y0, y1]]
    point: [x, y]
    -> bool
    """
    return (bbox[0][0] <= point[0] <= bbox[0][1] and
            bbox[1][0] <= point[1] <= bbox[1][1])


bbox_cache_ = {}  # Should help speed up ATLAS membership checking.


def make_radio_combination_signature(radio_annotation, wcs, radio_positions,
                                     subject, pix_offset, radio_survey='atlas'):
    """Generates a unique signature for a radio annotation.

    radio_annotation: 'radio' dictionary from a classification.
    wcs: World coordinate system associated with the radio image.
    radio_positions: [[RA, DEC]] NumPy array.
    subject: RGZ subject dict.
    pix_offset: (x, y) pixel position of this radio subject on the radio image.
    radio_survey: 'atlas' or 'first'.
    -> Something immutable
    """
    from . import rgz_data as data
    # My choice of immutable object will be stringified crowdastro radio
    # indices.
    zooniverse_id = subject['zooniverse_id']
    if radio_survey == 'atlas':
        subject_fits = data.get_radio_fits(subject)
        subject_wcs = astropy.wcs.WCS(subject_fits.header)
    elif radio_survey == 'first':
        subject_wcs = wcs  # FIRST wcs is taken straight from the RGZ FITS file.

    radio_ids = []
    x_offset, y_offset = pix_offset
    for c in radio_annotation.values():
        # Note that the x scale is not the same as the IR scale, but the scale
        # factor is included in the annotation, so I have multiplied this out
        # here for consistency.
        scale_width = c.get('scale_width', '')
        scale_height = c.get('scale_height', '')
        if scale_width:
            scale_width = float(scale_width)
        else:
            # Sometimes, there's no scale, so I've included a default scale.
            scale_width = config['surveys'][radio_survey]['scale_width']

        if scale_height:
            scale_height = float(scale_height)
        else:
            scale_height = config['surveys'][radio_survey]['scale_height']

        # These numbers are in terms of the PNG images, so I need to multiply by
        # the click-to-fits ratio.
        scale_width *= config['surveys'][radio_survey]['click_to_fits_x']
        scale_height *= config['surveys'][radio_survey]['click_to_fits_y']

        subject_bbox = [
            [
                float(c['xmin']) * scale_width,
                float(c['xmax']) * scale_width,
            ],
            [
                float(c['ymin']) * scale_height,
                float(c['ymax']) * scale_height,
            ],
        ]

        # ...and by the mosaic ratio. There's probably double-up here, but this
        # makes more sense.
        scale_width *= config['surveys'][radio_survey]['mosaic_scale_x']
        scale_height *= config['surveys'][radio_survey]['mosaic_scale_y']
        # Get the bounding box of the radio source in pixels.
        # Format: [xs, ys]
        bbox = [
            [
                float(c['xmin']) * scale_width,
                float(c['xmax']) * scale_width,
            ],
            [
                float(c['ymin']) * scale_height,
                float(c['ymax']) * scale_height,
            ],
        ]
        assert bbox[0][0] < bbox[0][1]
        assert bbox[1][0] < bbox[1][1]

        # Convert the bounding box into RA/DEC.
        bbox = wcs.all_pix2world(bbox[0] + x_offset, bbox[1] + y_offset,
                                 FITS_CONVENTION)
        subject_bbox = subject_wcs.all_pix2world(subject_bbox[0],
                subject_bbox[1], FITS_CONVENTION)
        # TODO(MatthewJA): Remove (or disable) this sanity check.

        # The bbox is backwards along the x-axis for some reason.
        bbox[0] = bbox[0][::-1]
        assert bbox[0][0] < bbox[0][1]
        assert bbox[1][0] < bbox[1][1]

        bbox = numpy.array(bbox)

        # What is this radio source called? Check if we have an object in the
        # bounding box. We'll cache these results because there is a lot of
        # overlap.
        cache_key = tuple(tuple(b) for b in bbox)
        if cache_key in bbox_cache_:
            index = bbox_cache_[cache_key]
        else:
            x_gt_min = radio_positions[:, 0] >= bbox[0, 0]
            x_lt_max = radio_positions[:, 0] <= bbox[0, 1]
            y_gt_min = radio_positions[:, 1] >= bbox[1, 0]
            y_lt_max = radio_positions[:, 1] <= bbox[1, 1]
            within = numpy.all([x_gt_min, x_lt_max, y_gt_min, y_lt_max], axis=0)
            indices = numpy.where(within)[0]

            if len(indices) == 0:
                logging.debug('Skipping radio source not in catalogue for '
                              '%s', zooniverse_id)
                continue
            else:
                if len(indices) > 1:
                    logging.debug('Found multiple (%d) radio matches '
                                  'for %s', len(indices), zooniverse_id)

                index = indices[0]

            bbox_cache_[cache_key] = index

        radio_ids.append(str(index))

    logging.debug('Sorting radio IDs...')
    radio_ids.sort()
    logging.debug('Sorted radio IDs.')

    if not radio_ids:
        raise CatalogueError('No catalogued radio sources.')

    return ';'.join(radio_ids)


def parse_classification(classification, subject, radio_positions, wcs,
                         pix_offset, radio_survey='atlas'):
    """Converts a raw RGZ classification into a classification dict.

    Scales all positions and flips y axis of clicks.

    classification: RGZ classification dict.
    subject: Associated RGZ subject dict.
    radio_positions: [[RA, DEC]] NumPy array.
    wcs: World coordinate system of the radio image.
    pix_offset: (x, y) pixel position of this radio subject on the radio image.
    radio_survey: 'atlas' or 'first'.
    -> dict mapping radio signature to list of corresponding IR host pixel
        locations.
    """
    result = {}

    n_invalid = 0

    for annotation in classification['annotations']:
        if 'radio' not in annotation:
            # This is a metadata annotation and we can ignore it.
            continue

        if annotation['radio'] == 'No Contours':
            # I'm not sure how this occurs. I'm going to ignore it.
            continue

        try:
            radio_signature = make_radio_combination_signature(
                    annotation['radio'], wcs, radio_positions,
                    subject, pix_offset, radio_survey=radio_survey)
        except CatalogueError:
            # Ignore invalid annotations.
            n_invalid += 1
            logging.debug('Ignoring invalid annotation for %s.',
                          subject['zooniverse_id'])
            continue

        ir_locations = []
        if annotation['ir'] != 'No Sources':
            for ir_click in annotation['ir']:
                ir_x = float(annotation['ir'][ir_click]['x'])
                ir_y = float(annotation['ir'][ir_click]['y'])

                # Rescale to a consistent size.
                ir_x *= config['surveys'][radio_survey]['click_to_fits_x']
                ir_y *= config['surveys'][radio_survey]['click_to_fits_y']

                # Ignore out-of-range data.
                if not 0 <= ir_x <= config['surveys'][
                        radio_survey]['fits_width']:
                    n_invalid += 1
                    continue

                if not 0 <= ir_y <= config['surveys'][
                        radio_survey]['fits_height']:
                    n_invalid += 1
                    continue

                # Flip the y axis to match other data conventions.
                ir_y = config['surveys'][radio_survey]['fits_height'] - ir_y

                # Rescale to match the mosaic WCS.
                ir_x *= config['surveys'][radio_survey]['mosaic_scale_x']
                ir_y *= config['surveys'][radio_survey]['mosaic_scale_y']

                # Move to the reference location of the radio subject.
                ir_x += pix_offset[0]
                ir_y += pix_offset[1]

                # Convert the location into RA/DEC.
                (ir_x,), (ir_y,) = wcs.wcs_pix2world([ir_x], [ir_y], 1)

                ir_location = (ir_x, ir_y)
                ir_locations.append(ir_location)

            result[radio_signature] = ir_locations

    if n_invalid:
        logging.debug('%d invalid annotations for %s.', n_invalid,
                      subject['zooniverse_id'])

    return result


def import_classifications(f_h5, radio_survey='atlas', test=False):
    """Imports Radio Galaxy Zoo classifications into crowdastro.

    f_h5: An HDF5 file.
    radio_survey: 'atlas' or 'first'.
    test: Flag to run on only 10 subjects. Default False.
    """
    # TODO(MatthewJA): This only works for ATLAS/CDFS. Generalise.
    from . import rgz_data as data
    radio_prefix = (
        '/atlas/cdfs/' if radio_survey == 'atlas' else '/first/first/')
    radio_positions = f_h5[radio_prefix + 'numeric'][:, :2]
    radio_ids = f_h5[radio_prefix + 'string']['zooniverse_id']
    classification_positions = []
    classification_combinations = []
    classification_usernames = []

    if radio_survey == 'atlas':
        with astropy.io.fits.open(
                # RGZ only has cdfs classifications
                config['data_sources']['atlas_cdfs_image'],
                ignore_blank=True) as atlas_image:
            wcs = astropy.wcs.WCS(atlas_image[0].header).dropaxis(3).dropaxis(2)

    for obj_index, radio_id in enumerate(radio_ids):
        subject = data.get_subject(radio_id.decode('ascii'))
        assert subject['zooniverse_id'] == radio_ids[obj_index].decode('ascii')

        classifications = data.get_subject_classifications(subject)

        if radio_survey == 'atlas':
            offset, = wcs.all_world2pix([subject['coords']], FITS_CONVENTION)
        elif radio_survey == 'first':
            try:
                path = os.path.join(
                        config['data_sources']['first_images'],
                        '{}.fits'.format(subject['metadata']['source']))
                with astropy.io.fits.open(path,
                        ignore_blank=True) as first_image:
                    # RGZ images only have the (two) spatial axes.
                    wcs = astropy.wcs.WCS(first_image[0].header)
                    offset, = wcs.all_world2pix([subject['coords']],
                                                FITS_CONVENTION)
            except FileNotFoundError:
                logging.warning('Could not find FIRST image: {}'.format(path))
                continue

        # The coords are of the middle of the subject.
        offset[0] -= (config['surveys'][radio_survey]['fits_width'] *
                      config['surveys'][radio_survey]['mosaic_scale_x'] // 2)
        offset[1] -= (config['surveys'][radio_survey]['fits_height'] *
                      config['surveys'][radio_survey]['mosaic_scale_y'] // 2)

        for c_index, classification in enumerate(classifications):
            user_name = classification.get('user_name', '').encode(
                    'ascii', errors='ignore')
            # Usernames actually don't have an upper length limit on RGZ(?!) so
            # I'll cap everything at 50 characters for my own sanity.
            if len(user_name) > 50:
                user_name = user_name[:50]

            classification = parse_classification(classification, subject,
                                                  radio_positions, wcs, offset,
                                                  radio_survey=radio_survey)
            full_radio = '|'.join(classification.keys())
            for radio, locations in classification.items():
                if not locations:
                    locations = [(None, None)]

                for click_index, location in enumerate(locations):
                    # Check whether the click index is 0 to maintain the
                    # assumption that we only need the first click.
                    pos_row = (obj_index, location[0], location[1],
                               click_index == 0)
                    com_row = (obj_index, full_radio, radio)
                    # A little redundancy here with the index, but we can assert
                    # that they are the same later to check integrity.
                    classification_positions.append(pos_row)
                    classification_combinations.append(com_row)
                    classification_usernames.append(user_name)

    combinations_dtype = [('index', 'int'),
                          ('full_signature', '<S{}'.format(
                                    MAX_RADIO_SIGNATURE_LENGTH)),
                          ('signature', '<S{}'.format(
                                    MAX_RADIO_SIGNATURE_LENGTH))]
    classification_positions = numpy.array(classification_positions,
                                           dtype=float)
    classification_combinations = numpy.array(classification_combinations,
                                              dtype=combinations_dtype)

    f_h5[radio_prefix].create_dataset('classification_positions',
                                      data=classification_positions,
                                      dtype=float)
    f_h5[radio_prefix].create_dataset('classification_usernames',
                                      data=classification_usernames,
                                      dtype='<S50')
    f_h5[radio_prefix].create_dataset('classification_combinations',
                                      data=classification_combinations,
                                      dtype=combinations_dtype)


def _populate_parser(parser):
    parser.description = 'Imports and standardises data into crowdastro.'
    parser.add_argument('--h5', default='data/crowdastro.h5',
                        help='HDF5 output file')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Run with a small number of subjects',)
    parser.add_argument('--ir', choices={'swire', 'wise'},
                        default='swire', help='which infrared survey to use')


def check_raw_data():
    """Validates existence and correctness of raw data files."""
    for source, filename in config['data_sources'].items():
        if source == 'radio_galaxy_zoo_db':
            # Skip the MongoDB name.
            continue

        if not os.path.exists(filename):
            logging.error(
                    '{} expected at {} but not found'.format(source, filename))

        if source in config['data_checksums']:
            valid = checksum_file(filename, config['data_checksums'][source])
            if not valid:
                logging.error('{} has incorrect hash'.format(filename))
            else:
                logging.debug('{} has correct hash'.format(filename))


def _main(args):
    check_raw_data()
    with h5py.File(args.h5, 'w') as f_h5:
        prep_h5(f_h5, args.ir)
        import_first(f_h5, test=args.test)
        import_atlas(f_h5, test=args.test, field='cdfs')
        import_atlas(f_h5, test=args.test, field='elais')
        if args.ir == 'swire':
            import_swire(f_h5, field='cdfs')
            import_swire(f_h5, field='elais')
        elif args.ir == 'wise':
            import_wise(f_h5, radio_survey='first')
            import_wise(f_h5, radio_survey='atlas', field='cdfs')
            import_wise(f_h5, radio_survey='atlas', field='elais')
        import_norris(f_h5)
        import_fan(f_h5)
        import_classifications(f_h5, radio_survey='atlas')
        import_classifications(f_h5, radio_survey='first')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
