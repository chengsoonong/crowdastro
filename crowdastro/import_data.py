"""Imports and standardises data into crowdastro.

Matthew Alger
The Australian National University
2016
"""

import argparse
import csv
import logging
import warnings

from astropy.coordinates import SkyCoord
import astropy.io.fits
import astropy.utils.exceptions
import astropy.wcs
import h5py
import numpy
import scipy.spatial.distance
import sklearn.neighbors

from .config import config
from .exceptions import CatalogueError

VERSION = '0.5.1'  # Data version, not module version!
MAX_RADIO_SIGNATURE_LENGTH = 50  # max number of components * individual
                                 # component signature size.
MAX_NAME_LENGTH = 50  # b
MAX_ZOONIVERSE_ID_LENGTH = 20  # b
PATCH_RADIUS = config['patch_radius']  # px
ARCMIN = 1 / 60  # deg
CANDIDATE_RADIUS = ARCMIN  # deg
FITS_CONVENTION = 1

def prep_h5(f_h5, ir_survey):
    """Creates hierarchy in HDF5 file."""
    f_h5.create_group('/atlas/cdfs')
    f_h5.create_group('/{}/cdfs'.format(ir_survey))
    f_h5.attrs['version'] = VERSION
    f_h5.attrs['ir_survey'] = ir_survey


def import_atlas(f_h5, test=False):
    """Imports the ATLAS dataset into crowdastro, as well as associated SWIRE.

    f_h5: An HDF5 file.
    test: Flag to run on only 10 subjects. Default False.
    """
    from . import rgz_data as data

    # Fetch groups from HDF5.
    cdfs = f_h5['/atlas/cdfs']

    # First pass, I'll find coords, names, and Zooniverse IDs, as well as how
    # many data points there are.

    coords = []
    names = []
    zooniverse_ids = []

    # We need the ATLAS name, but we can only get it by going through the
    # ATLAS catalogue and finding the nearest component.
    # https://github.com/chengsoonong/crowdastro/issues/63
    # Fortunately, @jbanfield has already done this, so we can just load
    # that CSV and match the names.
    # TODO(MatthewJA): This matches the ATLAS component ID, but maybe we should\
    # be using the name instead.
    rgz_to_atlas = {}
    with open(config['data_sources']['rgz_to_atlas']) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rgz_to_atlas[row['ID_RGZ']] = row['ID']

    all_subjects = data.get_all_subjects(survey='atlas', field='cdfs')
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
    # RA, DEC, radio, (distance to SWIRE object added later)
    dim = (n_cdfs, 1 + 1 + image_size)
    numeric = cdfs.create_dataset('_numeric', shape=dim, dtype='float32')

    # Load image patches and store numeric data.
    with astropy.io.fits.open(config['data_sources']['atlas_image'],
                              ignore_blank=True) as atlas_image:
        wcs = astropy.wcs.WCS(atlas_image[0].header).dropaxis(3).dropaxis(2)
        pix_coords = wcs.all_world2pix(coords, FITS_CONVENTION)
        assert pix_coords.shape[1] == 2
        logging.debug('Fetching %d ATLAS images.', len(pix_coords))
        for index, (x, y) in enumerate(pix_coords):
            radio = atlas_image[0].data[0, 0,  # stokes, freq
                    int(y) - config['surveys']['atlas']['fits_height'] // 2 :
                    int(y) + config['surveys']['atlas']['fits_height'] // 2 ,
                    int(x) - config['surveys']['atlas']['fits_width'] // 2 :
                    int(x) + config['surveys']['atlas']['fits_width'] // 2 ]
            numeric[index, 0] = coords[index][0]
            numeric[index, 1] = coords[index][1]
            numeric[index, 2 : 2 + image_size] = radio.reshape(-1)

    logging.debug('ATLAS imported.')

    # TODO(MatthewJA): Partition into training/testing sets, ideally using
    # expert or gold standard classifications.


def remove_nulls(n):
    """Swaps nulls with zeros."""
    if n == 'null':
        return 0

    return n


def import_swire(f_h5):
    """Imports the SWIRE dataset into crowdastro.

    f_h5: An HDF5 file.
    """
    names = []
    rows = []
    logging.debug('Reading SWIRE catalogue.')
    with open(config['data_sources']['swire_catalogue']) as f_tbl:
        # This isn't a valid ASCII table, so Astropy can't handle it. This means
        # we have to parse it manually.
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
    atlas_positions = f_h5['/atlas/cdfs/_numeric'][:, :2]
    logging.debug('Computing SWIRE k-d tree.')
    swire_tree = sklearn.neighbors.KDTree(swire_positions, metric='euclidean')
    indices = numpy.concatenate(
            swire_tree.query_radius(atlas_positions, CANDIDATE_RADIUS))
    indices = numpy.unique(indices)

    logging.debug('Found %d SWIRE objects near ATLAS objects.', len(indices))

    names = names[indices]
    rows = rows[indices]
    swire_positions = swire_positions[indices]

    # Get distances.
    logging.debug('Finding ATLAS-SWIRE object distances.')
    distances = scipy.spatial.distance.cdist(atlas_positions, swire_positions,
                                             'euclidean')
    assert distances.shape[0] == atlas_positions.shape[0]
    assert distances.shape[1] == swire_positions.shape[0]
    logging.debug('Done finding distances.')

    # Write numeric data to HDF5.
    rows[:, 8] = distances.min(axis=0)
    atlas_numeric = f_h5['/atlas/cdfs/_numeric']
    f_h5['/atlas/cdfs'].create_dataset('numeric', dtype='float32',
            shape=(atlas_numeric.shape[0],
                   atlas_numeric.shape[1] + len(indices)))
    f_h5['/atlas/cdfs/numeric'][:, :atlas_numeric.shape[1]] = atlas_numeric
    f_h5['/atlas/cdfs/numeric'][:, atlas_numeric.shape[1]:] = distances

    del f_h5['/atlas/cdfs/_numeric']

    image_size = (PATCH_RADIUS * 2) ** 2
    dim = (rows.shape[0], rows.shape[1] + image_size)
    numeric = f_h5['/swire/cdfs'].create_dataset('numeric', shape=dim,
                                                 dtype='float32')
    numeric[:, :rows.shape[1]] = rows
    f_h5['/swire/cdfs'].create_dataset('string', data=names)

    # Load and store radio images.
    logging.debug('Importing radio patches.')
    with astropy.io.fits.open(config['data_sources']['atlas_image'],
                              ignore_blank=True) as atlas_image:
        wcs = astropy.wcs.WCS(atlas_image[0].header).dropaxis(3).dropaxis(2)
        pix_coords = wcs.all_world2pix(swire_positions, FITS_CONVENTION)
        assert pix_coords.shape[1] == 2
        assert pix_coords.shape[0] == len(indices)
        logging.debug('Fetching %d ATLAS patches.', len(indices))

        for index, (x, y) in enumerate(pix_coords):
            radio = atlas_image[0].data[0, 0,  # stokes, freq
                    int(y) - PATCH_RADIUS :
                    int(y) + PATCH_RADIUS ,
                    int(x) - PATCH_RADIUS :
                    int(x) + PATCH_RADIUS ]
            numeric[index, -image_size:] = radio.reshape(-1)


def import_wise(f_h5):
    """Imports the WISE dataset into crowdastro.

    f_h5: An HDF5 file.
    """
    names = []
    rows = []
    logging.debug('Reading WISE catalogue.')
    with open(config['data_sources']['wise_catalogue']) as f_tbl:
        # This isn't a valid ASCII table, so Astropy can't handle it. This means
        # we have to parse it manually.
        for _ in range(105):  # Skip the first 105 lines.
            next(f_tbl)

        # Get the column names.
        columns = [c.strip() for c in next(f_tbl).strip().split('|')][1:-1]
        assert len(columns) == 45

        for _ in range(3):  # Skip the next three lines.
            next(f_tbl)

        for row in f_tbl:
            row = row.strip().split()
            assert len(row) == 45
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

    logging.debug('Found %d WISE objects.', len(names))

    # Sort by name.
    rows_to_names = dict(zip(rows, names))
    rows.sort(key=rows_to_names.get)
    names.sort()

    names = numpy.array(names, dtype='<S{}'.format(MAX_NAME_LENGTH))
    rows = numpy.array(rows)

    # Filter on distance - only include image data for WISE objects within a
    # given radius of an ATLAS object. Otherwise, there's way too much data to
    # store.
    wise_positions = rows[:, :2]
    atlas_positions = f_h5['/atlas/cdfs/_numeric'][:, :2]
    logging.debug('Computing WISE k-d tree.')
    wise_tree = sklearn.neighbors.KDTree(wise_positions, metric='euclidean')
    indices = numpy.concatenate(
            wise_tree.query_radius(atlas_positions, CANDIDATE_RADIUS))
    indices = numpy.unique(indices)

    logging.debug('Found %d WISE objects near ATLAS objects.', len(indices))

    names = names[indices]
    rows = rows[indices]
    wise_positions = wise_positions[indices]

    # Get distances.
    logging.debug('Finding ATLAS-WISE object distances.')
    distances = scipy.spatial.distance.cdist(atlas_positions, wise_positions,
                                             'euclidean')
    assert distances.shape[0] == atlas_positions.shape[0]
    assert distances.shape[1] == wise_positions.shape[0]
    logging.debug('Done finding distances.')

    # Write numeric data to HDF5.
    rows[:, 6] = distances.min(axis=0)
    atlas_numeric = f_h5['/atlas/cdfs/_numeric']
    f_h5['/atlas/cdfs'].create_dataset('numeric', dtype='float32',
            shape=(atlas_numeric.shape[0],
                   atlas_numeric.shape[1] + len(indices)))
    f_h5['/atlas/cdfs/numeric'][:, :atlas_numeric.shape[1]] = atlas_numeric
    f_h5['/atlas/cdfs/numeric'][:, atlas_numeric.shape[1]:] = distances

    del f_h5['/atlas/cdfs/_numeric']

    image_size = (PATCH_RADIUS * 2) ** 2
    dim = (rows.shape[0], rows.shape[1] + image_size)
    numeric = f_h5['/wise/cdfs'].create_dataset('numeric', shape=dim,
                                                 dtype='float32')
    numeric[:, :rows.shape[1]] = rows
    f_h5['/wise/cdfs'].create_dataset('string', data=names)

    # Load and store radio images.
    logging.debug('Importing radio patches.')
    with astropy.io.fits.open(config['data_sources']['atlas_image'],
                              ignore_blank=True) as atlas_image:
        wcs = astropy.wcs.WCS(atlas_image[0].header).dropaxis(3).dropaxis(2)
        pix_coords = wcs.all_world2pix(wise_positions, FITS_CONVENTION)
        assert pix_coords.shape[1] == 2
        assert pix_coords.shape[0] == len(indices)
        logging.debug('Fetching %d ATLAS patches.', len(indices))

        for index, (x, y) in enumerate(pix_coords):
            radio = atlas_image[0].data[0, 0,  # stokes, freq
                    int(y) - PATCH_RADIUS :
                    int(y) + PATCH_RADIUS ,
                    int(x) - PATCH_RADIUS :
                    int(x) + PATCH_RADIUS ]
            numeric[index, -image_size:] = radio.reshape(-1)


def import_norris(f_h5):
    """Imports the Norris et al. (2006) labels.

    f_h5: crowdastro HDF5 file with WISE or SWIRE already imported.
    """
    ir_survey = f_h5.attrs['ir_survey']
    ir_names = f_h5['/{}/cdfs/string'.format(ir_survey)]
    ir_positions = f_h5['/{}/cdfs/numeric'.format(ir_survey)][:, :2]
    ir_tree = sklearn.neighbors.KDTree(ir_positions)
    with open(config['data_sources']['norris_coords'], 'r') as norris_dat:
        norris_coords = [r.strip().split('|') for r in norris_dat]
    norris_labels = numpy.zeros((ir_positions.shape[0],))
    for ra, dec in norris_coords:
        # Find a neighbour.
        skycoord = SkyCoord(ra=ra, dec=dec, unit=('hourangle', 'deg'))
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


def make_radio_combination_signature(radio_annotation, wcs, atlas_positions,
                                     subject, pix_offset):
    """Generates a unique signature for a radio annotation.

    radio_annotation: 'radio' dictionary from a classification.
    wcs: World coordinate system associated with the ATLAS image.
    atlas_positions: [[RA, DEC]] NumPy array.
    subject: RGZ subject dict.
    pix_offset: (x, y) pixel position of this radio subject on the ATLAS image.
    -> Something immutable
    """
    from . import rgz_data as data
    # TODO(MatthewJA): This only works on ATLAS. Generalise.
    # My choice of immutable object will be stringified crowdastro ATLAS
    # indices.
    zooniverse_id = subject['zooniverse_id']
    subject_fits = data.get_radio_fits(subject)
    subject_wcs = astropy.wcs.WCS(subject_fits.header)

    atlas_ids = []
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
            scale_width = config['surveys']['atlas']['scale_width']

        if scale_height:
            scale_height = float(scale_height)
        else:
            scale_height = config['surveys']['atlas']['scale_height']

        # These numbers are in terms of the PNG images, so I need to multiply by
        # the click-to-fits ratio.
        scale_width *= config['surveys']['atlas']['click_to_fits_x']
        scale_height *= config['surveys']['atlas']['click_to_fits_y']

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
        scale_width *= config['surveys']['atlas']['mosaic_scale_x']
        scale_height *= config['surveys']['atlas']['mosaic_scale_y']
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
            x_gt_min = atlas_positions[:, 0] >= bbox[0, 0]
            x_lt_max = atlas_positions[:, 0] <= bbox[0, 1]
            y_gt_min = atlas_positions[:, 1] >= bbox[1, 0]
            y_lt_max = atlas_positions[:, 1] <= bbox[1, 1]
            within = numpy.all([x_gt_min, x_lt_max, y_gt_min, y_lt_max], axis=0)
            indices = numpy.where(within)[0]

            if len(indices) == 0:
                logging.debug('Skipping radio source not in catalogue for '
                              '%s', zooniverse_id)
                continue
            else:
                if len(indices) > 1:
                    logging.debug('Found multiple (%d) ATLAS matches '
                                  'for %s', len(indices), zooniverse_id)

                index = indices[0]

            bbox_cache_[cache_key] = index

        atlas_ids.append(str(index))

    atlas_ids.sort()

    if not atlas_ids:
        raise CatalogueError('No catalogued radio sources.')

    return ';'.join(atlas_ids)


def parse_classification(classification, subject, atlas_positions, wcs,
                         pix_offset):
    """Converts a raw RGZ classification into a classification dict.

    Scales all positions and flips y axis of clicks.

    classification: RGZ classification dict.
    subject: Associated RGZ subject dict.
    atlas_positions: [[RA, DEC]] NumPy array.
    wcs: World coordinate system of the ATLAS image.
    pix_offset: (x, y) pixel position of this radio subject on the ATLAS image.
    -> dict mapping radio signature to corresponding IR host pixel location
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
                    annotation['radio'], wcs, atlas_positions,
                    subject, pix_offset)
        except CatalogueError:
            # Ignore invalid annotations.
            n_invalid += 1
            logging.debug('Ignoring invalid annotation for %s.',
                          subject['zooniverse_id'])
            continue

        if annotation['ir'] == 'No Sources':
            ir_location = (None, None)
        else:
            ir_x = float(annotation['ir']['0']['x'])
            ir_y = float(annotation['ir']['0']['y'])

            # Rescale to a consistent size.
            ir_x *= config['surveys']['atlas']['click_to_fits_x']
            ir_y *= config['surveys']['atlas']['click_to_fits_y']

            # Ignore out-of-range data.
            if not 0 <= ir_x <= config['surveys']['atlas']['fits_width']:
                n_invalid += 1
                continue

            if not 0 <= ir_y <= config['surveys']['atlas']['fits_height']:
                n_invalid += 1
                continue

            # Flip the y axis to match other data conventions.
            ir_y = config['surveys']['atlas']['fits_height'] - ir_y

            # Rescale to match the mosaic WCS.
            ir_x *= config['surveys']['atlas']['mosaic_scale_x']
            ir_y *= config['surveys']['atlas']['mosaic_scale_y']

            # Move to the reference location of the radio subject.
            ir_x += pix_offset[0]
            ir_y += pix_offset[1]

            # Convert the location into RA/DEC.
            (ir_x,), (ir_y,) = wcs.wcs_pix2world([ir_x], [ir_y], 1)

            ir_location = (ir_x, ir_y)

        result[radio_signature] = ir_location

    if n_invalid:
        logging.debug('%d invalid annotations for %s.', n_invalid,
                      subject['zooniverse_id'])

    return result


def import_classifications(f_h5, test=False):
    """Imports Radio Galaxy Zoo classifications into crowdastro.

    f_h5: An HDF5 file.
    test: Flag to run on only 10 subjects. Default False.
    """
    # TODO(MatthewJA): This only works for ATLAS/CDFS. Generalise.
    from . import rgz_data as data
    atlas_positions = f_h5['/atlas/cdfs/numeric'][:, :2]
    atlas_ids = f_h5['/atlas/cdfs/string']['zooniverse_id']
    classification_positions = []
    classification_combinations = []
    classification_usernames = []

    with astropy.io.fits.open(config['data_sources']['atlas_image'],
                              ignore_blank=True) as atlas_image:
        wcs = astropy.wcs.WCS(atlas_image[0].header).dropaxis(3).dropaxis(2)

    for obj_index, atlas_id in enumerate(atlas_ids):
        subject = data.get_subject(atlas_id.decode('ascii'))
        assert subject['zooniverse_id'] == atlas_ids[obj_index].decode('ascii')
        classifications = data.get_subject_classifications(subject)
        offset, = wcs.all_world2pix([subject['coords']], FITS_CONVENTION)
        # The coords are of the middle of the subject.
        offset[0] -= (config['surveys']['atlas']['fits_width'] *
                      config['surveys']['atlas']['mosaic_scale_x'] // 2)
        offset[1] -= (config['surveys']['atlas']['fits_height'] *
                      config['surveys']['atlas']['mosaic_scale_y'] // 2)

        for c_index, classification in enumerate(classifications):
            user_name = classification.get('user_name', '').encode(
                    'ascii', errors='ignore')
            # Usernames actually don't have an upper length limit on RGZ(?!) so
            # I'll cap everything at 50 characters for my own sanity.
            if len(user_name) > 50:
                user_name = user_name[:50]

            classification = parse_classification(classification, subject,
                                                  atlas_positions, wcs, offset)
            full_radio = '|'.join(classification.keys())
            for radio, location in classification.items():
                pos_row = (obj_index, location[0], location[1])
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

    f_h5['/atlas/cdfs/'].create_dataset('classification_positions',
                                        data=classification_positions,
                                        dtype=float)
    f_h5['/atlas/cdfs/'].create_dataset('classification_usernames',
                                        data=classification_usernames,
                                        dtype='<S50')
    f_h5['/atlas/cdfs/'].create_dataset('classification_combinations',
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


def _main(args):
    with h5py.File(args.h5, 'w') as f_h5:
        prep_h5(f_h5, args.ir)
        import_atlas(f_h5, test=args.test)
        if args.ir == 'swire':
            import_swire(f_h5)
        elif args.ir == 'wise':
            import_wise(f_h5)
        import_norris(f_h5)
        import_fan(f_h5)
        import_classifications(f_h5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    _populate_parser(parser)
    args = parser.parse_args()
    _main(args)
