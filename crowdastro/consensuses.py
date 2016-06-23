"""Finds the consensus crowd classifications in Radio Galaxy Zoo.

Matthew Alger
The Australian National University
2016
"""

import argparse
import collections
import csv
import itertools
import logging

import h5py
import numpy
import scipy.linalg.basic
import scipy.ndimage.filters
import scipy.ndimage.morphology
import scipy.stats
import sklearn.mixture
import sklearn.neighbors


MAX_RADIO_SIGNATURE_LENGTH = 50  # max number of components * individual
                                 # component signature size.


def maybe_mean(arr):
    arr = arr[~numpy.all(numpy.isnan(arr), axis=1)]
    if arr.shape[0] == 0:
        return (float('nan'), float('nan'))

    return arr.mean(axis=0)


def pg_means(points, significance=0.01, projections=24):
    """Find a consensus location with the PG-means algorithm.

    points: Array of points with dimension (N, 2).
    significance: Optional. Significance level for increasing the Gaussian
        count.
    projections: Optional. How many projections to try before accepting.
    -> (x, y), boolean of whether PG-means succeeded.
    """
    k = 1

    last_gmm = None
    while True:
        # Fit a Gaussian mixture model with k components.
        gmm = sklearn.mixture.GMM(n_components=k, covariance_type='full')
        try:
            gmm.fit(points)
        except ValueError:
            if last_gmm is None:
                return maybe_mean(points), False

            return last_gmm.means_[last_gmm.weights_.argmax()], False
        last_gmm = gmm

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
            return gmm.means_[gmm.weights_.argmax()], True

        k += 1


def kde(points):
    """Find a consensus location with the KDE algorithm.

    points: [[x, y]] NumPy array.
    -> (x, y) consensus location, boolean of whether KDE succeeded
    """
    if len(points) == 1:
        return points.mean(axis=0), False

    if len(points) == 0:
        return (float('nan'), float('nan')), False

    X, Y = numpy.mgrid[0:200, 0:200]
    positions = numpy.vstack([X.ravel(), Y.ravel()])
    try:
        kernel = scipy.stats.gaussian_kde(points.T)
    except scipy.linalg.basic.LinAlgError:
        logging.warning('LinAlgError in KD estimation.')
        return maybe_mean(points), False
    except ValueError as e:
        logging.warning('ValueError in KD estimation: %s', e.message)
        return maybe_mean(points), False

    kp = kernel(positions)
    if numpy.isnan(kp).sum() > 0:
        logging.warning('NaN in KD estimation.')
        return maybe_mean(points), False

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
    return (x_peak, y_peak), True


def find_consensuses(f_h5, ir_survey):
    """Find Radio Galaxy Zoo crowd consensuses.

    f_h5: crowdastro HDF5 file.
    ir_survey: SWIRE or WISE.
    """
    if 'consensus_objects' in f_h5['/atlas/cdfs/']:
        del f_h5['/atlas/cdfs/consensus_objects']

    class_positions = f_h5['/atlas/cdfs/classification_positions']
    class_combinations = f_h5['/atlas/cdfs/classification_combinations']
    assert len(class_positions) == len(class_combinations)

    logging.debug('Finding consensuses for %d classifications.',
                  len(class_combinations))

    # Pre-build the IR tree.
    ir_coords = f_h5['/{}/cdfs/numeric'.format(ir_survey)][:, :2]
    ir_tree = sklearn.neighbors.KDTree(ir_coords)

    cons_positions = []
    cons_combinations = []

    # Data integrity and assumptions checks.
    assert numpy.array_equal(class_positions[:, 0],
                             class_combinations['index'])
    assert numpy.array_equal(class_positions[:, 0],
                             sorted(class_positions[:, 0]))

    pos_groups = itertools.groupby(class_positions, key=lambda z: z[0])
    com_groups = itertools.groupby(class_combinations, key=lambda z: z['index'])

    for (i, pos_group), (j, com_group) in zip(pos_groups, com_groups):
        assert i == j

        com_group = list(com_group)  # For multiple iterations.
        pos_group = list(pos_group)
        total_classifications = 0

        # Find the radio consensus. Be wary when counting: If there are multiple
        # AGNs identified in one subject, *that classification will appear
        # multiple times*. I'm going to deal with this by dividing the weight of
        # each classification by how many pipes it contains plus one.
        radio_counts = {}  # Radio signature -> Count
        for _, full_com, _ in com_group:
            count = radio_counts.get(full_com, 0)
            count += 1 / (full_com.count(b'|') + 1)
            radio_counts[full_com] = count

            total_classifications += 1 / (full_com.count(b'|') + 1)

        for count in radio_counts.values():
            # Despite the divisions, we should end up with integers overall.
            assert numpy.isclose(round(count), count)
        assert numpy.isclose(round(total_classifications),
                             total_classifications)

        radio_consensus = max(radio_counts, key=radio_counts.get)

        # Find the location consensus. For each radio combination, run a
        # location consensus function on the positions associated with that
        # combination.
        for radio_signature in radio_consensus.split(b'|'):
            percentage_consensus = (radio_counts[radio_consensus] /
                                    total_classifications)
            locations = []
            for (_, x, y), (_, full, radio) in zip(pos_group, com_group):
                if full == radio_consensus and radio == radio_signature:
                    locations.append((x, y))
            locations = numpy.array(locations)
            locations = locations[~numpy.all(numpy.isnan(locations), axis=1)]
            (x, y), success = pg_means(locations)

            if numpy.isnan(x) or numpy.isnan(y):
                logging.debug('Skipping NaN PG-means output.')
                continue

            # Match the (x, y) position to an IR object.
            dist, ind = ir_tree.query([(x, y)])

            # TODO(MatthewJA): Cut-off based on distance.
            # Since IR data is sorted, we can deal directly with indices.
            cons_positions.append((i, ind[0][0], success))
            cons_combinations.append((i, radio_signature, percentage_consensus))

    logging.debug('Found %d consensuses (before duplicate removal).',
                  len(cons_positions))

    # Remove duplicates. For training data, I don't really care if radio
    # combinations overlap (though I need to care if I generate a catalogue!) so
    # just take duplicated locations and pick the one with the highest radio
    # consensus that has success.
    cons_objects = {}  # Maps IR index to (ATLAS index, success,
                       #                   percentage_consensus)
    for (atlas_i, ir_i, success), (atlas_j, radio, percentage) in zip(
            cons_positions, cons_combinations):
        assert atlas_i == atlas_j

        if ir_i not in cons_objects:
            cons_objects[ir_i] = (atlas_i, success, percentage)
            continue

        if cons_objects[ir_i][1] and not success:
            # Preference successful KDE/PG-means.
            continue

        if not cons_objects[ir_i][1] and success:
            # Preference successful KDE/PG-means.
            cons_objects[ir_i] = (atlas_i, success, percentage)
            continue

        # If we get this far, we have the same success state. Choose based on
        # radio consensus.
        if percentage > cons_objects[ir_i][2]:
            cons_objects[ir_i] = (atlas_i, success, percentage)
            continue

    logging.debug('Found %d consensuses.', int(len(cons_objects)))

    cons_objects = numpy.array([(atlas_i, ir_i, success, percentage)
            for ir_i, (atlas_i, success, percentage)
            in sorted(cons_objects.items())])

    # Write rows to file.
    cons_objects = numpy.array(cons_objects, dtype=float)
    f_h5['/atlas/cdfs/'].create_dataset('consensus_objects',
                                        data=cons_objects, dtype=float)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', default='crowdastro.h5',
                        help='HDF5 IO file')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)

    with h5py.File(args.h5, 'r+') as f_h5:
        assert f_h5.attrs['version'] == '0.5.0'
        ir_survey = f_h5.attrs['ir_survey']
        find_consensuses(f_h5, ir_survey)
