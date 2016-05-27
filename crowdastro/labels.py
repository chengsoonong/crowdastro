"""Utilities for manipulating labels from the Radio Galaxy Zoo."""

import collections
import functools
import logging
import operator
import sqlite3
import struct
import sys

import astropy.coordinates
import astropy.wcs
import numpy
import scipy.ndimage.morphology
import scipy.stats
import sklearn.mixture

from . import config
from . import data
from .rgz_analysis import consensus

atlas_catalogue_cache = {}
with open(config.get('atlas_catalogue_path')) as f:
    atlas_catalogue = [l.split() for l in f if not l.startswith('#')]

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
        gmm = sklearn.mixture.GMM(n_components=k, covariance_type='full')
        try:
            gmm.fit(points)
        except ValueError:
            return None

        for _ in range(projections):
            # Project the data to one dimension.
            projection_vector = numpy.random.random(size=(2,))
            projected_points = numpy.dot(points, projection_vector)
            # Project the model to one dimension. We need the CDF in one
            # dimension, so we'll sample some data points and project them.
            n_samples = 1000
            samples = numpy.dot(gmm.sample(n_samples), projection_vector)
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

def get_subject_consensus_kde(subject, conn, table):
    """Finds the volunteer consensus for radio combination and source location.

    subject: RGZ subject dict.
    conn: SQLite3 database connection.
    table: Name of table in database containing frozen classifications.
    -> (dict mapping radio signatures to ((x, y) NumPy arrays, or None),
        percentage agreement on radio combination,
        dict mapping radio signatures to boolean agreement on location)
    """
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = ('SELECT full_radio_signature, part_radio_signature, source_x, '
          'source_y FROM classifications WHERE zooniverse_id=?')

    classifications = list(cur.execute(sql, [str(subject['zooniverse_id'])]))
    if not classifications:
        return {}, 0, {}

    frs_counter = collections.Counter([c['full_radio_signature']
                                       for c in classifications])
    most_common_frs = frs_counter.most_common(1)[0][0]
    radio_consensus_classifications = collections.defaultdict(list)

    for classification in classifications:
        if classification['full_radio_signature'] == most_common_frs:
            radio_consensus_classifications[
                    classification['part_radio_signature']
            ].append((classification['source_x'], classification['source_y']))

    logging.debug('Most common radio signature for %s: %s',
                  subject['zooniverse_id'], most_common_frs)

    radio_agreement = frs_counter.most_common(1)[0][1] / len(classifications)

    consensus = {}  # Maps radio signatures to (x, y) NumPy arrays.
    location_agreement = {}  # Maps radio signatures to agreement percentages.
    for radio_signature in radio_consensus_classifications:
        n_no_source = 0  # Number of people who think there is no source.
        xs = []
        ys = []
        for c in radio_consensus_classifications[radio_signature]:
            if c[0] is None or c[1] is None:
                # No source.
                n_no_source += 1
                continue

            raise NotImplementedError('Scales have been changed and code not '
                                      'updated.')
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
            agreement = n_no_source / len(
                    radio_consensus_classifications[radio_signature])
            location_agreement[radio_signature] = agreement
            continue

        # Find the consensus source.
        X, Y = numpy.mgrid[0:200, 0:200]
        positions = numpy.vstack([X.ravel(), Y.ravel()])
        points = numpy.vstack([xs, ys])
        try:
            kernel = scipy.stats.gaussian_kde(points)
        except scipy.linalg.basic.LinAlgError:
            logging.debug('LinAlgError in KD estimation.')
            continue
        except ValueError:
            logging.debug('ValueError in KD estimation.')
            continue

        kp = kernel(positions)
        if numpy.isnan(kp).sum() > 0:
            consensus[radio_signature] = points.mean(axis=1)
            location_agreement[radio_signature] = 0
            continue

        neighborhood = numpy.ones((10, 10))
        Z = numpy.reshape(kp.T, X.shape)
        local_max = scipy.ndimage.filters.maximum_filter(
                Z, footprint = neighborhood) == Z
        background = (Z == 0)
        eroded_background = scipy.ndimage.morphology.binary_erosion(
                background, structure = neighborhood, border_value=1)
        detected_peaks = local_max ^ eroded_background

        x_peak = float(X[Z == Z.max()][0])
        y_peak = float(Y[Z == Z.max()][0])
        consensus[radio_signature] = (x_peak, y_peak)
        location_agreement[radio_signature] = 1

    return consensus, radio_agreement, location_agreement

def get_subject_consensus_pg_means(subject, conn, table, significance=0.02):
    """Finds the volunteer consensus for radio combination and source location.

    subject: RGZ subject dict.
    conn: SQLite3 database connection.
    table: Name of table in database containing frozen classifications.
    significance: Optional. Significance level for splitting consensus coords.
    -> (dict mapping radio signatures to ((x, y) NumPy arrays, or None),
        percentage agreement on radio combination,
        dict mapping radio signatures to percentage agreement on location)
    """
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    sql = ('SELECT full_radio_signature, part_radio_signature, source_x, '
          'source_y FROM classifications WHERE zooniverse_id=?')

    classifications = list(cur.execute(sql, [str(subject['zooniverse_id'])]))
    if not classifications:
        return {}, 0, {}

    frs_counter = collections.Counter([c['full_radio_signature']
                                       for c in classifications])
    most_common_frs = frs_counter.most_common(1)[0][0]
    radio_consensus_classifications = collections.defaultdict(list)

    for classification in classifications:
        if classification['full_radio_signature'] == most_common_frs:
            radio_consensus_classifications[
                    classification['part_radio_signature']
            ].append((classification['source_x'], classification['source_y']))

    logging.debug('Most common radio signature for %s: %s',
                  subject['zooniverse_id'], most_common_frs)

    radio_agreement = frs_counter.most_common(1)[0][1] / len(classifications)

    consensus = {}  # Maps radio signatures to (x, y) NumPy arrays.
    location_agreement = {}  # Maps radio signatures to agreement percentages.
    for radio_signature in radio_consensus_classifications:
        n_no_source = 0  # Number of people who think there is no source.
        xs = []
        ys = []
        for c in radio_consensus_classifications[radio_signature]:
            if c[0] is None or c[1] is None:
                # No source.
                n_no_source += 1
                continue

            raise NotImplementedError('Scales have been changed and code not '
                                      'updated.')
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
            agreement = n_no_source / len(
                    radio_consensus_classifications[radio_signature])
            location_agreement[radio_signature] = agreement
            continue

        # Find the consensus source.
        points = numpy.vstack([xs, ys])
        gmm = pg_means(points.T, significance=significance, projections=24)

        if gmm is None:
            # In case of no agreement, assume we have no source.
            # TODO(MatthewJA): Kyle treats this situation by using the average
            # location. I'm not sure how valid this is but I should do something
            # similar. At any rate I should probably do more than return None.
            logging.warning('No consensus for %s but non-zero classifications.',
                            subject['zooniverse_id'])
            consensus[radio_signature] = numpy.array([None, None])
            location_agreement[radio_signature] = 0
        else:
            consensus[radio_signature] = gmm.means_[gmm.weights_.argmax()]
            classes = gmm.predict(points.T)
            agreements = sum(classes == gmm.weights_.argmax())
            agreement = agreements / len(classes)
            location_agreement[radio_signature] = agreement

    return consensus, radio_agreement, location_agreement

def freeze_classifications(db_path, table, atlas=False):
    """Freezes Radio Galaxy Zoo classifications into a SQLite database.

    Warning: table argument is not validated! This could be dangerous.

    db_path: Path to SQLite database. If this doesn't exist, it will be created.
    table: Name of table to freeze classifications into. If this exists, it will
        be cleared.
    atlas: Whether to only freeze ATLAS subjects. Default False (though this
        function currently only works for True).
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS {}'.format(table))
    conn.commit()

    c.execute('CREATE TABLE {} (zooniverse_id TEXT, full_radio_signature TEXT, '
              'part_radio_signature TEXT, source_x REAL, source_y REAL'
              ')'.format(table))
    conn.commit()

    sql = ('INSERT INTO {} (zooniverse_id, full_radio_signature, '
           'part_radio_signature, source_x, source_y) VALUES '
           '(?, ?, ?, ?, ?)'.format(table))

    def iter_sql_params():
        n_subjects = data.get_all_subjects(atlas=atlas).count()
        for idx, subject in enumerate(data.get_all_subjects(atlas=atlas)):
            print('Freezing consensus. Progress: {} ({:.02%})'.format(
                    idx, idx / n_subjects), file=sys.stderr, end='\r')

            zooniverse_id = subject['zooniverse_id']
            for c in data.get_subject_classifications(subject):
                radio_locations = parse_classification(c, subject)

                # Different spacer (i.e. not semicolon) stops collisions.
                full_radio = '|'.join(sorted(radio_locations.keys()))

                for radio, location in radio_locations.items():
                    if location is None:
                        x = None
                        y = None
                    else:
                        x, y = location

                    yield (zooniverse_id, full_radio, radio, x, y)

    c.executemany(sql, iter_sql_params())

    conn.commit()

def freeze_consensuses(db_path, consensus_table, atlas=False):
    """Freezes Radio Galaxy Zoo consensuses into a SQLite database.

    Warning: table arguments are not validated! This could be dangerous.

    db_path: Path to SQLite database.
    consensus_table: Name of table to freeze consensuses into. If this exists,
        it will be cleared.
    atlas: Whether to only freeze ATLAS subjects. Default False.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS {}'.format(consensus_table))
    conn.commit()

    c.execute('CREATE TABLE {} (zooniverse_id TEXT, radio_signature TEXT, '
              'source_x REAL, source_y REAL, radio_agreement REAL, '
              'location_agreement REAL)'.format(consensus_table))
    conn.commit()

    sql = ('INSERT INTO {} (zooniverse_id, radio_signature, source_x, '
           'source_y, radio_agreement, location_agreement) '
           'VALUES (?, ?, ?, ?, ?, ?)'.format(consensus_table))

    params = []

    n_subjects = data.get_all_subjects(atlas=atlas).count()
    for idx, subject in enumerate(data.get_all_subjects(atlas=atlas)):
        if idx % 500 == 0:
            logging.debug('Executing %d queries.', len(params))
            c.executemany(sql, params)
            conn.commit()
            params = []

        print('Freezing consensus. Progress: {} ({:.02%})'.format(
                idx, idx / n_subjects), file=sys.stderr, end='\r')

        zooniverse_id = subject['zooniverse_id']
        cons, radio_agreement, location_agreement = (
                get_subject_consensus_kde(subject, conn, consensus_table))
        for radio_signature, (x, y) in cons.items():
            params.append((zooniverse_id, radio_signature, x, y,
                           radio_agreement,
                           location_agreement[radio_signature]))
    print()

    c.executemany(sql, params)
    conn.commit()
