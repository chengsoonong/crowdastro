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
    X, Y = numpy.mgrid[0:200, 0:200]
    positions = numpy.vstack([X.ravel(), Y.ravel()])
    try:
        kernel = scipy.stats.gaussian_kde(points.T)
    except scipy.linalg.basic.LinAlgError:
        logging.debug('LinAlgError in KD estimation.')
        return maybe_mean(points), False
    except ValueError:
        logging.debug('ValueError in KD estimation.')
        return maybe_mean(points), False

    kp = kernel(positions)
    if numpy.isnan(kp).sum() > 0:
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


def find_consensuses(f_h5, f_csv):
    """Find Radio Galaxy Zoo crowd consensuses.

    f_h5: crowdastro HDF5 file.
    """
    if 'consensus_positions' in f_h5['/atlas/cdfs/']:
        del f_h5['/atlas/cdfs/consensus_positions']

    if 'consensus_combinations' in f_h5['/atlas/cdfs/']:
        del f_h5['/atlas/cdfs/consensus_combinations']

    class_positions = f_h5['/atlas/cdfs/classification_positions']
    class_combinations = f_h5['/atlas/cdfs/classification_combinations']

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

        # Find the radio consensus. Be wary when counting: If there are multiple
        # AGNs identified in one subject, *that classification will appear
        # multiple times*. I'm going to deal with this by dividing the weight of
        # each classification by how many pipes it contains plus one.
        radio_counts = {}  # Radio signature -> Count
        for _, full_com, _ in com_group:
            count = radio_counts.get(full_com, 0)
            count += 1 / (full_com.count(b'|') + 1)
            radio_counts[full_com] = count

        for count in radio_counts.values():
            # Despite the divisions, we should end up with integers overall.
            assert numpy.isclose(round(count), count)

        radio_consensus = max(radio_counts, key=radio_counts.get)

        # Find the location consensus. For each radio combination, run a
        # location consensus function on the positions associated with that
        # combination.
        for radio_signature in radio_consensus.split(b'|'):
            locations = []
            for (_, x, y), (_, full, radio) in zip(pos_group, com_group):
                if full == radio_consensus and radio == radio_signature:
                    locations.append((x, y))
            locations = numpy.array(locations)
            (x, y), success = pg_means(locations)

            cons_positions.append((i, x, y, success))
            cons_combinations.append((i, radio_signature))

    # Write rows to file.
    combinations_dtype = [('index', int),
                          ('signature', '<S{}'.format(
                                    MAX_RADIO_SIGNATURE_LENGTH))]
    cons_positions = numpy.array(cons_positions, dtype=float)
    cons_combinations = numpy.array(cons_combinations, dtype=combinations_dtype)
    f_h5['/atlas/cdfs/'].create_dataset('consensus_positions',
                                        data=cons_positions, dtype=float)
    f_h5['/atlas/cdfs'].create_dataset('consensus_combinations',
                                       data=cons_combinations,
                                       dtype=combinations_dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', default='crowdastro.h5',
                        help='HDF5 IO file')
    parser.add_argument('--csv', default='crowdastro.csv',
                        help='CSV input file')
    args = parser.parse_args()

    with h5py.File(args.h5, 'r+') as f_h5:
        with open(args.csv, 'r') as f_csv:
            find_consensuses(f_h5, f_csv)
