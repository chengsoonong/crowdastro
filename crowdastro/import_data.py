"""Imports and standardises data into crowdastro.

Matthew Alger
The Australian National University
2016
"""

import argparse
import csv
import logging

import astropy.wcs
import h5py
import numpy
import sklearn.neighbors

from . import rgz_data as data
from .config import config
from .exceptions import CatalogueError

VERSION = '0.3.0'  # Data version, not module version!
MAX_RADIO_SIGNATURE_LENGTH = 50  # max number of components * individual
                                 # component signature size.
ARCMIN = 1 / 60


def prep_h5(f_h5):
    """Creates hierarchy in HDF5 file."""
    cdfs = f_h5.create_group('/atlas/cdfs')
    swire_cdfs = f_h5.create_group('/swire/cdfs')
    f_h5.attrs['version'] = VERSION


def prep_csv(f_csv):
    """Writes headers of CSV."""
    writer = csv.writer(f_csv)
    writer.writerow(['index', 'survey', 'field', 'zooniverse_id', 'name',
                     'header'])


def import_atlas(f_h5, f_csv, test=False):
    """Imports the ATLAS dataset into crowdastro, as well as associated SWIRE.

    f_h5: An HDF5 file.
    f_csv: A CSV file.
    test: Flag to run on only 10 subjects. Default False.
    """
    # Fetch groups from HDF5.
    cdfs = f_h5['/atlas/cdfs']
    swire_cdfs = f_h5['/swire/cdfs']

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

    # Store coords in HDF5.
    coords = numpy.array(coords)
    coords_ds = cdfs.create_dataset('positions', data=coords)

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

    headers = []

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

        # TODO(MatthewJA): Remove dependency on FITS images.
        radio_5x5_fits = data.get_radio_fits(subject, size='5x5')
        header = radio_5x5_fits.header.tostring()
        headers.append(header)

    # Store Zooniverse IDs, names, and headers in CSV.
    writer = csv.writer(f_csv)
    for index, (zooniverse_id, name, header) in enumerate(
            zip(zooniverse_ids, names, headers)):
        writer.writerow([index, 'atlas', 'cdfs', zooniverse_id, name, header])

    # Finally, partition training/testing/validation data sets.
    n_data = len(zooniverse_ids)
    indices = numpy.arange(n_data, dtype='int')
    numpy.random.shuffle(indices)

    test_max = int(config['test_size'] * n_data)
    n_training = int((1 - config['test_size']) * n_data)
    validation_max = int(config['validation_size'] * n_training)
    testing_indices = indices[:test_max]
    validation_indices = indices[test_max:validation_max]
    training_indices = indices[validation_max:]

    cdfs.create_dataset('testing_indices', data=testing_indices)
    cdfs.create_dataset('validation_indices', data=validation_indices)
    cdfs.create_dataset('training_indices', data=training_indices)


def remove_nulls(n):
    """Swaps nulls with zeros."""
    if n == 'null':
        return 0

    return n


def import_swire(f_h5, f_csv):
    """Imports the SWIRE dataset into crowdastro.

    f_h5: An HDF5 file.
    f_csv: A CSV file.
    """
    names = []
    rows = []
    with open(config['data_sources']['swire_catalogue']) as f_tbl:
        # This isn't a valid ASCII table, so Astropy can't handle it.
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
            # Extra -1 is so we can store ATLAS subject indices later.
            # Extra 0s are so we can store the train/test/validation set.
            rows.append((ra, dec, flux_ap2_36, flux_ap2_45, flux_ap2_58,
                         flux_ap2_80, flux_ap2_24, stell_36, -1, 0, 0, 0))
            names.append(name)

    # Sort by name.
    rows_to_names = dict(zip(rows, names))
    rows.sort(key=rows_to_names.get)
    names.sort()

    # Find SWIRE objects that are within range of an ATLAS subject, and assign
    # them to an ATLAS subject and a corresponding train/test/validation index.
    # Only commit SWIRE objects within range.
    rows = numpy.array(rows)
    positions = rows[:, :2]
    swire_tree = sklearn.neighbors.KDTree(positions, metric='chebyshev')
    seen = set()  # SWIRE objects we've already seen (to avoid reassignments).
    atlas_train = set(f_h5['/atlas/cdfs/training_indices'])
    atlas_test = set(f_h5['/atlas/cdfs/testing_indices'])
    atlas_valid = set(f_h5['/atlas/cdfs/validation_indices'])
    for index, atlas_pos in enumerate(f_h5['/atlas/cdfs/positions']):
        neighbours = swire_tree.query_radius([atlas_pos], ARCMIN)[0]
        for neighbour in neighbours:
            if neighbour in seen:
                continue

            seen.add(neighbour)
            rows[neighbour, 8] = index
            if index in atlas_train:
                rows[neighbour, 9] = 1
            elif index in atlas_valid:
                rows[neighbour, 10] = 1
            elif index in atlas_test:
                rows[neighbour, 11] = 1

    write_names = []
    write_rows = []
    for index, name in enumerate(names):
        if index in seen:
            write_names.append(name)
            row = rows[index]
            write_rows.append(row)
    write_rows = numpy.array(write_rows)

    assert len(write_rows) == len(write_names)
    logging.debug('Found %d SWIRE objects near an ATLAS subject.', len(rows))

    # Write names to CSV.
    writer = csv.writer(f_csv)
    for index, name in enumerate(write_names):
        writer.writerow([index, 'swire', '', '', name, ''])

    # Write numeric data to HDF5.
    f_h5['/swire/cdfs'].create_dataset('catalogue', data=write_rows)


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
                                     zooniverse_id=None):
    """Generates a unique signature for a radio annotation.

    radio_annotation: 'radio' dictionary from a classification.
    wcs: World coordinate system associated with this classification. Generate
        this using astropy.wcs.WCS(fits_header).
    atlas_positions: [[x, y]] NumPy array.
    zooniverse_id: Zooniverse ID (for logging). Optional.
    -> Something immutable
    """
    # TODO(MatthewJA): This only works on ATLAS. Generalise.
    # My choice of immutable object will be stringified crowdastro ATLAS
    # indices.
    atlas_ids = []
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
        bbox = wcs.wcs_pix2world(bbox[0], bbox[1], 1)

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
                if zooniverse_id:
                    logging.debug('Skipping radio source not in catalogue for '
                                  '%s', zooniverse_id)
                else:
                    logging.debug('Skipping radio source not in catalogue.')
                continue
            else:
                if len(indices) > 1:
                    if zooniverse_id:
                        logging.debug('Found multiple (%d) ATLAS matches '
                                      'for %s', len(indices), zooniverse_id)
                    else:
                        logging.debug('Found multiple (%d) ATLAS matches',
                                      len(indices))

                index = indices[0]

            bbox_cache_[cache_key] = index

        atlas_ids.append(str(index))

    atlas_ids.sort()

    if not atlas_ids:
        raise CatalogueError('No catalogued radio sources.')

    return ';'.join(atlas_ids)


def parse_classification(classification, subject, atlas_positions):
    """Converts a raw RGZ classification into a classification dict.

    Scales all positions and flips y axis of clicks.

    classification: RGZ classification dict.
    subject: Associated RGZ subject dict.
    atlas_positions: [[x, y]] NumPy array.
    -> dict mapping radio signature to corresponding IR host pixel location
    """
    result = {}

    fits = data.get_radio_fits(subject)
    wcs = astropy.wcs.WCS(fits.header)

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
                    zooniverse_id=subject['zooniverse_id'])
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

            # Convert the location into RA/DEC.
            (ir_x,), (ir_y,) = wcs.wcs_pix2world([ir_x], [ir_y], 1)

            ir_location = (ir_x, ir_y)

        result[radio_signature] = ir_location

    if n_invalid:
        logging.debug('%d invalid annotations for %s.', n_invalid,
                      subject['zooniverse_id'])

    return result


def import_classifications(f_h5, f_csv):
    """Imports Radio Galaxy Zoo classifications into crowdastro.

    f_h5: An HDF5 file.
    f_csv: A CSV file.
    """
    # TODO(MatthewJA): This only works for ATLAS/CDFS. Generalise.
    reader = csv.DictReader(f_csv)

    atlas_positions = f_h5['/atlas/cdfs/positions']
    classification_positions = []
    classification_combinations = []
    for obj_index, obj in enumerate(reader):
        if obj['survey'] != 'atlas':
            continue

        assert obj['field'] == 'cdfs'

        subject = data.get_subject(obj['zooniverse_id'])
        classifications = data.get_subject_classifications(subject)
        for c_index, classification in enumerate(classifications):
            classification = parse_classification(classification, subject,
                                                  atlas_positions)
            full_radio = '|'.join(classification.keys())
            for radio, location in classification.items():
                pos_row = (int(obj['index']), location[0], location[1])
                com_row = (int(obj['index']), full_radio, radio)
                # A little redundancy here with the index, but we can assert
                # that they are the same later to check integrity.
                classification_positions.append(pos_row)
                classification_combinations.append(com_row)

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
    f_h5['/atlas/cdfs/'].create_dataset('classification_combinations',
                                        data=classification_combinations,
                                        dtype=combinations_dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', default='crowdastro.h5',
                        help='HDF5 output file')
    parser.add_argument('--csv', default='crowdastro.csv',
                        help='CSV output file')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Run with a small number of subjects',)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)

    with h5py.File(args.h5, 'w') as f_h5:
        with open(args.csv, 'w') as f_csv:
            prep_h5(f_h5)
            prep_csv(f_csv)
            import_atlas(f_h5, f_csv, test=args.test)
            import_swire(f_h5, f_csv)

        with open(args.csv, 'r') as f_csv:
            # Classifications shouldn't modify the CSV.
            import_classifications(f_h5, f_csv)
