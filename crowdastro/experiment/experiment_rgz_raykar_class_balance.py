"""Tests the effect of class imbalance on the Raykar algorithm applied to RGZ.

Matthew Alger
The Australian National University
2016
"""

import argparse
import collections
import logging

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy
import sklearn
import sklearn.decomposition
import sklearn.metrics

from . import runners
from .experiment_rgz_raykar import top_n_accurate_targets
from .results import Results
from .. import __version__
from ..crowd.util import majority_vote
from ..crowd.raykar import RaykarClassifier
from ..plot import vertical_scatter_ba


def main(crowdastro_h5_path, training_h5_path, results_h5_path,
         overwrite=False, plot=False, n_annotators=10):
    with h5py.File(crowdastro_h5_path, 'r') as crowdastro_h5, \
            h5py.File(training_h5_path, 'r') as training_h5:

        n_splits = crowdastro_h5['/wise/cdfs/test_sets'].shape[0]
        n_examples, n_params = training_h5['features'].shape
        n_params += 1  # Bias term.
        n_params += 1  # Number of annotators.
        n_params += crowdastro_h5['/wise/cdfs/rgz_raw_labels'].shape[0] * 2
        methods = [
            'Downsampled negatives',
            'No resampling',
        ]
        model = '{} crowdastro.crowd.raykar.RaykarClassifier, '.format(
                    __version__)

        results = Results(results_h5_path, methods, n_splits, n_examples,
                          n_params, model)

        features = training_h5['features'].value
        targets = top_n_accurate_targets(crowdastro_h5,
                                         n_annotators=n_annotators)

        alphas_all_trials = {
            'Downsampled negatives': [],
            'No resampling': [],
        }
        betas_all_trials = {
            'Downsampled negatives': [],
            'No resampling': [],
        }
        for split_id, test_set in enumerate(
                    crowdastro_h5['/wise/cdfs/test_sets']):
            logging.info('Test {}/{}'.format(split_id + 1, n_splits))
            for method_id, method in enumerate(methods):
                logging.info('Method {} ({}/{})'.format(method, method_id + 1,
                                                        len(methods)))
                downsample = method == 'Downsampled negatives'
                runners.raykar(results, method, split_id, features,
                               targets, list(test_set),
                               overwrite=overwrite, n_restarts=1,
                               downsample=downsample)
                model = results.get_model(method, split_id)
                rc = RaykarClassifier.unserialise(model)
                logging.info('{} alpha: {}'.format(method, rc.a_))
                logging.info('{} beta: {}'.format(method, rc.b_))
                alphas_all_trials[method].append(rc.a_)
                betas_all_trials[method].append(rc.b_)

        for method in methods:
            alphas = numpy.mean(alphas_all_trials[method], axis=0)
            betas = numpy.mean(betas_all_trials[method], axis=0)
            logging.info('Average alphas for {}: {}'.format(method, alphas))
            logging.info('Average betas for {}: {}'.format(method, betas))

        if plot:
            matplotlib.rcParams['font.family'] = 'serif'
            matplotlib.rcParams['font.serif'] = ['Palatino Linotype']
            vertical_scatter_ba(
                    results,
                    crowdastro_h5['/wise/cdfs/norris_labels'].value,
                    rotation='horizontal')
            plt.ylim((0, 1))
            plt.show()

            to_hist = []
            for method in methods:
                alphas = numpy.mean(alphas_all_trials[method], axis=0)
                to_hist.append(alphas)
            to_hist = numpy.vstack(to_hist).T
            plt.hist(to_hist)
            plt.legend(methods)
            plt.xlabel('$\\alpha$')
            plt.ylabel('Number of labellers')
            plt.show()

            to_hist = []
            for method in methods:
                betas = numpy.mean(betas_all_trials[method], axis=0)
                to_hist.append(betas)
            to_hist = numpy.vstack(to_hist).T
            plt.hist(to_hist)
            plt.legend(methods)
            plt.xlabel('$\\beta$')
            plt.ylabel('Number of labellers')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crowdastro', default='data/crowdastro.h5',
                        help='HDF5 crowdastro data file')
    parser.add_argument('--training', default='data/training.h5',
                        help='HDF5 training data file')
    parser.add_argument(
        '--results', default='data/results_rgz_raykar_class_balance.h5',
        help='HDF5 results data file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--annotators', type=int, help='Number of annotators',
                        default=10)
    parser.add_argument('--plot', action='store_true', help='Generate a plot')
    args = parser.parse_args()

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    main(args.crowdastro, args.training, args.results, overwrite=args.overwrite,
         plot=args.plot, n_annotators=args.annotators)
