"""Runs the Raykar algorithm on a classification task with class imbalance.

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
import sklearn.cross_validation
import sklearn.datasets

from . import runners
from .. import __version__
from ..crowd.raykar import RaykarClassifier
from ..crowd.util import crowd_label
from ..plot import vertical_scatter_ba
from .results import Results


def main(results_h5_path, overwrite=False, plot=False, n_trials=5,
         n_examples=1000, n_dim=5, n_labellers=10, seed=0):
    numpy.random.seed(seed)
    methods = [
        '4:1',
        '2:1',
        '1:1',
        '1:2',
        '1:4',
    ]
    mr_methods = [[int(i) for i in m.split(':')] for m in methods]
    n_params = n_dim + 1 + 1 + n_labellers * 2
    crowd_alphas = numpy.random.uniform(0.5, 0.9, size=(n_labellers,))
    crowd_betas = numpy.random.uniform(0.5, 0.9, size=(n_labellers,))
    model = '{} crowdastro.raykar.RaykarClassifier'.format(__version__)
    results = Results(results_h5_path, methods, n_trials, n_examples,
                      n_params, model)

    alphas = collections.defaultdict(list)
    betas = collections.defaultdict(list)
    for trial in range(n_trials):
        for mr_method, method in zip(mr_methods, methods):
            # Generate some toy data.
            x, y = sklearn.datasets.make_classification(
                n_samples=n_examples,
                n_features=n_dim,
                n_informative=n_dim - (n_dim // 2),
                n_redundant=n_dim // 2,
                n_classes=2,
                weights=mr_method,
                flip_y=0.01,
                class_sep=1.0,
                shuffle=True,
                random_state=seed + trial)
            _, test_indices = sklearn.cross_validation.train_test_split(
                range(y.shape[0]), stratify=y, random_state=seed + trial,
                test_size=0.2)
            test_indices.sort()
            labels = crowd_label(y, crowd_alphas, crowd_betas)
            runners.raykar(results, method, trial, x, labels, test_indices,
                           overwrite=overwrite, downsample=False)
            model = results.get_model(method, trial)
            rc = RaykarClassifier.unserialise(model)
            alphas[method].extend(rc.a_)
            betas[method].extend(rc.b_)

    if plot:
        # violinplot takes a list of lists of observations.
        alpha_data = [alphas[m] for m in methods]
        beta_data = [betas[m] for m in methods]

        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Palatino Linotype']

        plt.subplot(1, 2, 1)
        plt.violinplot(alpha_data, showmeans=True)
        plt.ylim((0, 1))
        plt.ylabel('$\\alpha$')
        plt.xticks([1 + i for i in range(len(methods))], methods)
        plt.subplot(1, 2, 2)
        plt.violinplot(beta_data, showmeans=True)
        plt.ylim((0, 1))
        plt.ylabel('$\\beta$')
        plt.xticks([1 + i for i in range(len(methods))], methods)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',
                        default='data/results_raykar_class_balance.h5',
                        help='HDF5 results data file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--trials', type=int, help='Number of trials to run',
                        default=5)
    parser.add_argument('--plot', action='store_true', help='Generate a plot')
    args = parser.parse_args()

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    main(args.results, overwrite=args.overwrite, plot=args.plot,
         n_trials=args.trials)
