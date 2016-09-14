"""Runs the Yan algorithm on a simulated crowd classification task.

Matthew Alger
The Australian National University
2016
"""

import argparse
import collections
import csv
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy
import sklearn
import sklearn.cluster
import sklearn.cross_validation

from . import runners
from .. import __version__
from ..crowd.util import majority_vote
from ..plot import vertical_scatter_ba
from .results import Results


def main(input_csv_path, results_h5_path, overwrite=False, plot=False,
         seed=0, shuffle_seed=0):
    numpy.random.seed(seed)
    with open(input_csv_path, 'r') as f:
        reader = csv.reader(f)
        features = []
        labels = []
        for row in reader:
            label = row[-1] == '4'
            labels.append(label)
            feature = [float(i) if i != '?' else 0 for i in row[1:-1]]
            features.append(feature)
        features = numpy.array(features)
        labels = numpy.array(labels)

        n_splits = 3
        n_labellers = 5
        mask_rate = 0.5  # Lower = less masked.
        n_examples, n_params = features.shape
        n_params += 1  # Bias term.
        n_params += n_labellers * n_params  # w.
        methods = ['Yan', 'LR']
        model = '{} crowdastro.crowd.yan.YanClassifier,'.format(
                        __version__) + \
                '{} sklearn.linear_model.LogisticRegression'.format(
                        sklearn.__version__)

        results = Results(results_h5_path, methods, n_splits, n_examples,
                          n_params, model)

        # Generate the crowd labels. Cluster the data into T clusters and assign
        # each cluster to a labeller. That labeller is 100% accurate in that
        # cluster and 75% accurate everywhere else.
        km = sklearn.cluster.KMeans(n_clusters=n_labellers)
        km.fit(features)
        classes = km.predict(features)
        crowd_labels = numpy.tile(labels, (n_labellers, 1))
        for labeller in range(n_labellers):
            for i in range(n_examples):
                if classes[i] == labeller:
                    crowd_labels[labeller, i] = labels[i]
                elif numpy.random.random() < 0.25:
                    crowd_labels[labeller, i] = 1 - labels[i]
                else:
                    crowd_labels[labeller, i] = labels[i]
        # Randomly mask a percentage of the elements.
        mask = numpy.random.binomial(1, mask_rate, size=crowd_labels.shape)
        crowd_labels = numpy.ma.MaskedArray(crowd_labels, mask=mask)
        # Compute a majority vote of the crowd labels to use for LR.
        mv = majority_vote(crowd_labels)

        all_features = {
            'Yan': features,
            'LR': features,
        }
        targets = {
            'LR': mv,
            'Yan': crowd_labels,
        }

        ss = sklearn.cross_validation.ShuffleSplit(n_examples, n_iter=n_splits,
                test_size=0.25, random_state=shuffle_seed)
        for split_id, (train, test) in enumerate(ss):
            logging.info('Test {}/{}'.format(split_id + 1, n_splits))
            for method_id, method in enumerate(methods):
                logging.info('Method {} ({}/{})'.format(method, method_id + 1,
                                                        len(methods)))
                if method == 'LR':
                    runners.lr(results, method, split_id, all_features[method],
                               targets[method], sorted(test),
                               overwrite=overwrite)
                elif method == 'Yan':
                    runners.yan(results, method, split_id,
                                   all_features[method], targets[method],
                                   sorted(test), overwrite=overwrite,
                                   n_restarts=5)
                else:
                    raise ValueError('Unexpected method: {}'.format(method))

        if plot:
            matplotlib.rcParams['font.family'] = 'serif'
            matplotlib.rcParams['font.serif'] = ['Palatino Linotype']
            vertical_scatter_ba(results, labels, violin=False, minorticks=False)
            plt.ylim((0, 1))
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/breast_cancer_wisconsin.csv',
                        help='Input breast cancer data CSV')
    parser.add_argument('--results', default='data/results_yan.h5',
                        help='HDF5 results data file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--plot', action='store_true', help='Generate a plot')
    args = parser.parse_args()

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    main(args.input, args.results, overwrite=args.overwrite, plot=args.plot)
