"""Investigating the effect of label noise on the Raykar algorithm.

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
from ..crowd.util import balanced_accuracy, crowd_label
from ..plot import fillbetween, violinplot
from .results import Results


def main(results_h5_path, overwrite=False, plot=False, n_trials=5,
         n_examples=1000, n_dim=5, n_labellers=10, seed=0):
    numpy.random.seed(seed)
    methods = [
        'a = 0.2',
        'a = 0.3',
        'a = 0.4',
        'a = 0.5',
        'a = 0.6',
        'a = 0.7',
        'a = 0.8',
        'a = 0.9',
        'a = 1.0',
        'b = 0.2',
        'b = 0.3',
        'b = 0.4',
        'b = 0.5',
        'b = 0.6',
        'b = 0.7',
        'b = 0.8',
        'b = 0.9',
        'b = 1.0',
    ]
    # Convert the strings into test cases. This is a little easier to read than
    # writing tuples first and then generating strings and it works just as
    # well.
    mr_methods = []
    for m in methods:
        m = m.split(' = ')
        if m[0] == 'a':
            mr_methods.append((float(m[1]), 1.0))
        else:
            mr_methods.append((1.0, float(m[1])))

    n_params = n_dim + 1 + 1 + n_labellers * 2
    model = '{} crowdastro.raykar.RaykarClassifier'.format(__version__)
    results = Results(results_h5_path, methods, n_trials, n_examples,
                      n_params, model)

    # We'll store the plot results here.
    alpha_to_ests = collections.defaultdict(list)
    beta_to_ests = collections.defaultdict(list)
    alpha_to_bas = collections.defaultdict(list)
    beta_to_bas = collections.defaultdict(list)
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
            crowd_alphas = [mr_method[0]] * n_labellers
            crowd_betas = [mr_method[1]] * n_labellers
            labels = crowd_label(y, crowd_alphas, crowd_betas)
            runners.raykar(results, method, trial, x, labels, test_indices,
                           overwrite=overwrite, downsample=False)
            model = results.get_model(method, trial)
            rc = RaykarClassifier.unserialise(model)
            ba = balanced_accuracy(rc.predict(x[test_indices]),
                                   y[test_indices])

            if method.startswith('a'):
                alpha_to_ests[mr_method[0]].extend(rc.a_)
                alpha_to_bas[mr_method[0]].append(ba)
            else:
                beta_to_ests[mr_method[1]].extend(rc.b_)
                beta_to_bas[mr_method[1]].append(ba)

    if plot:
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['font.serif'] = ['Palatino Linotype']

        # alpha vs ba
        plt.subplot(2, 2, 1)
        fillbetween(
            sorted(alpha_to_bas),
            [alpha_to_bas[a] for a in sorted(alpha_to_bas)],
            facecolour='lightblue', edgecolour='blue')
        plt.ylim((0.5, 1.0))
        plt.ylabel('Balanced accuracy')

        # beta vs ba
        plt.subplot(2, 2, 2)
        fillbetween(
            sorted(beta_to_bas),
            [beta_to_bas[b] for b in sorted(beta_to_bas)],
            facecolour='lightblue', edgecolour='blue')
        plt.ylim((0.5, 1.0))
        plt.ylabel('Balanced accuracy')

        # alpha vs est alpha
        plt.subplot(2, 2, 3)
        fillbetween(
            sorted(alpha_to_ests),
            [alpha_to_ests[a] for a in sorted(alpha_to_ests)],
            facecolour='lightgreen', edgecolour='green')
        plt.xlabel('$\\alpha$')
        plt.ylabel('Estimated $\\alpha$')

        # beta vs est beta
        plt.subplot(2, 2, 4)
        fillbetween(
            sorted(beta_to_ests),
            [beta_to_ests[a] for a in sorted(beta_to_ests)],
            facecolour='lightgreen', edgecolour='green')
        plt.xlabel('$\\beta$')
        plt.ylabel('Estimated $\\beta$')

        # Space out the plots a little so the titles don't overlap.
        plt.subplots_adjust(wspace=0.25)

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',
                        default='data/results_raykar_noise.h5',
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
