"""Utilities for manipulating labels from the Radio Galaxy Zoo."""

import collections
import functools
import logging
import operator
import sqlite3
import struct
import sys

import numpy
import scipy.stats
import sklearn.mixture

from . import config
from . import data

DEFAULT_SCALE_WIDTH = 2.1144278606965172
FP_PRECISION = 14

def pg_means(points, significance=0.01, projections=24):
    """Cluster points with the PG-means algorithm.

    points: Array of points with dimension (2, N).
    significance: Optional. Significance level for increasing the Gaussian
        count.
    projections: Optional. How many projections to try before accepting.
    -> sklearn.mixture.GMM
    """
    k = 1
    
    while True:
        # Fit a Gaussian mixture model with k components.
        gmm = sklearn.mixture.GMM(n_components=k)
        try:
            gmm.fit(points)
        except ValueError:
            return None

        
        for _ in range(projections):
            # Project the data to one dimension.
            projection_vector = numpy.random.random(size=(2,))
            projected_points = points @ projection_vector
            # Project the model to one dimension. We need the CDF in one
            # dimension, so we'll sample some data points and project them.
            n_samples = 1000
            samples = gmm.sample(n_samples) @ projection_vector
            samples.sort()
            
            def cdf(x):
                for sample, y in zip(samples,
                                     numpy.arange(n_samples) / n_samples):
                    if sample >= x:
                        break
                return y
            
            _, p_value = scipy.stats.kstest(projected_points,
                                            numpy.vectorize(cdf))
            if p_value < significance:
                # Reject the null hypothesis.
                break
        else:
            # Null hypothesis was not broken.
            return gmm
        
        k += 1

def get_subject_consensus(subject, conn, table, significance=0.02):
    """Finds the volunteer consensus for radio combination and source location.

    subject: RGZ subject dict.
    conn: SQLite3 database connection.
    table: Name of table in database containing frozen classifications.
    significance: Optional. Significance level for splitting consensus coords.
    -> dict mapping radio signatures to ((x, y) NumPy arrays, or None).
    """
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = ('SELECT full_radio_signature, part_radio_signature, source_x, '
          'source_y FROM classifications WHERE subject_id=?')

    classifications = list(cur.execute(sql, [str(subject['_id'])]))
    if not classifications:
        return {}

    frs_counter = collections.Counter([c['full_radio_signature']
                                       for c in classifications])
    most_common_frs = frs_counter.most_common(1)[0][0]
    radio_consensus_classifications = collections.defaultdict(list)

    for classification in classifications:
        if classification['full_radio_signature'] == most_common_frs:
            radio_consensus_classifications[
                    classification['part_radio_signature']
            ].append((classification['source_x'], classification['source_y']))
    
    consensus = {}  # Maps radio signatures to (x, y) NumPy arrays.
    for radio_signature in radio_consensus_classifications:
        n_no_source = 0  # Number of people who think there is no source.
        xs = []
        ys = []
        for c in radio_consensus_classifications[radio_signature]:
            if c[0] is None or c[1] is None:
                # No source.
                n_no_source += 1
                continue

            x = c[0] * config.get('click_to_fits_x')
            y = c[1] * config.get('click_to_fits_y')
            xs.append(x)
            ys.append(y)

        if (n_no_source >
                len(radio_consensus_classifications[radio_signature]) // 2):
            # Majority think that there is no source.
            # Note that if half of people think there is no source and half
            # think that there is a source, we'll assume there is a source.
            consensus[radio_signature] = numpy.array([None, None])
            continue

        # Find the consensus source.
        points = numpy.vstack([xs, ys])
        gmm = pg_means(points.T, significance=significance, projections=24)

        if gmm is None:
            # In case of no agreement, assume we have no source.
            logging.warning('No consensus for %s but non-zero classifications.',
                            subject['zooniverse_id'])
            consensus[radio_signature] = numpy.array([None, None])
        else:
            consensus[radio_signature] = gmm.means_[gmm.weights_.argmax()]
    
    return consensus

def make_radio_combination_signature(radio_annotation):
    """Generates a unique signature for a radio annotation.
    
    radio_annotation: 'radio' dictionary from a classification.
    -> Something immutable
    """
    # My choice of immutable object will be a bytes object. This will be an
    # encoding of a tuple of the xmax values, sorted to ensure determinism, and
    # rounded to nix floating point errors.
    xmaxes = []
    for c in radio_annotation.values():
        # Note that the x scale is not the same as the IR scale, but the scale
        # factor is included in the annotation, so I have multiplied this out
        # here for consistency.
        scale_width = c.get('scale_width', '')
        if scale_width:
            scale_width = float(scale_width)
        else:
            # Sometimes, there's no scale, so I've included a default scale.
            scale_width = DEFAULT_SCALE_WIDTH

        # Keep everything within the maximum width, so that we can accurately
        # guess precision.
        scale_width /= config.get('click_image_width')

        xmax = round(float(c['xmax']) * scale_width, FP_PRECISION)

        # Some of the numbers are out of range. I don't have a good fallback, so
        # I'll just raise an error.
        if not 0 <= xmax <= 1:
            raise ValueError('Radio xmax out of range: {}'.format(xmax))
        
        xmaxes.append(xmax)

    xmaxes.sort()

    # Pack the floats to convert them into bytes. I will use little-endian,
    # 32-bit floats.
    xmaxes = [struct.pack('<f', xmax) for xmax in xmaxes]

    return functools.reduce(operator.add, xmaxes, b'')

def parse_classification(classification):
    """Converts a raw RGZ classification into a classification dict.

    classification: RGZ classification dict.
    -> (subject MongoDB ID,
        dict mapping radio signature to corresponding IR host pixel location)
    """
    result = {}

    for annotation in classification['annotations']:
        if 'radio' not in annotation:
            # This is a metadata annotation and we can ignore it.
            continue

        if annotation['radio'] == 'No Contours':
            # I'm not sure how this occurs. I'm going to ignore it.
            continue

        try:
            radio_signature = make_radio_combination_signature(
                    annotation['radio'])
        except ValueError:
            # Ignore invalid annotations.
            continue

        if annotation['ir'] == 'No Sources':
            ir_location = None
        else:
            ir_x = float(annotation['ir']['0']['x'])
            ir_y = float(annotation['ir']['0']['y'])

            # Ignore out-of-range data.
            if not 0 <= ir_x <= config.get('click_image_width'):
                continue

            if not 0 <= ir_y <= config.get('click_image_height'):
                continue

            ir_location = (ir_x, ir_y)

        result[radio_signature] = ir_location

    assert len(classification['subject_ids']) == 1

    return classification['subject_ids'][0], result

def freeze_classifications(db_path, table):
    """Freezes Radio Galaxy Zoo classifications into a SQLite database.

    Warning: table argument is not validated! This could be dangerous.

    db_path: Path to SQLite database. If this doesn't exist, it will be created.
    table: Name of table to freeze classifications into. If this exists, it will
        be cleared.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS {}'.format(table))
    conn.commit()

    c.execute('CREATE TABLE {} (subject_id TEXT, full_radio_signature BLOB, '
              'part_radio_signature BLOB, source_x REAL, source_y REAL'
              ')'.format(table))
    conn.commit()

    sql = ('INSERT INTO {} (subject_id, full_radio_signature, '
           'part_radio_signature, source_x, source_y) VALUES '
           '(?, ?, ?, ?, ?)'.format(table))

    def iter_sql_params():
        for c in data.get_all_classifications():
            subject_id, radio_locations = parse_classification(c)
            # We add the length to full_radio to help prevent collisions.
            full_radio = b''
            for k in sorted(radio_locations.keys()):
                full_radio += b'|' + k  # Spacer helps prevent collisions.

            for radio, location in radio_locations.items():
                if location is None:
                    x = None
                    y = None
                else:
                    x, y = location

                yield (str(subject_id), full_radio, radio, x, y)

    c.executemany(sql, iter_sql_params())

    conn.commit()

def freeze_consensuses(db_path, classification_table, consensus_table,
                       significance=0.02, atlas=False):
    """Freezes Radio Galaxy Zoo consensuses into a SQLite database.

    Warning: table arguments are not validated! This could be dangerous.

    db_path: Path to SQLite database.
    classification_table: Name of table containing frozen classifications.
    consensus_table: Name of table to freeze consensuses into. If this exists,
        it will be cleared.
    significance: Optional. Significance level for splitting consensus coords.
    atlas: Whether to only freeze ATLAS subjects. Default False.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS {}'.format(consensus_table))
    conn.commit()

    c.execute('CREATE TABLE {} (subject_id TEXT, radio_signature BLOB, '
              'source_x REAL, source_y REAL'
              ')'.format(consensus_table))
    conn.commit()

    sql = ('INSERT INTO {} (subject_id, radio_signature, source_x, source_y) '
           'VALUES (?, ?, ?, ?)'.format(consensus_table))

    params = []

    n_subjects = data.get_all_subjects(atlas=atlas).count()
    for idx, subject in enumerate(data.get_all_subjects(atlas=atlas)):
        if idx % 1000 == 0:
            c.executemany(sql, params)
            conn.commit()
            params = []

        print('Freezing consensus. Progress: {} ({:.02%})'.format(
                idx, idx / n_subjects), file=sys.stderr, end='\r')

        subject_id = str(subject['_id'])
        cons = get_subject_consensus(subject, conn, classification_table)
        for radio_signature, (x, y) in cons.items():
            params.append((subject_id, radio_signature, x, y))
    print()

    c.executemany(sql, params)
    conn.commit()
