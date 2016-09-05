"""Compares logistic regression to random forests, trained on Norris and Fan.

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

from . import runners
from .results import Results
from ..plot import vertical_scatter_ba


def main(crowdastro_h5_path, training_h5_path, results_h5_path,
         overwrite=False, plot=False):
    with h5py.File(crowdastro_h5_path, 'r') as crowdastro_h5, \
            h5py.File(training_h5_path, 'r') as training_h5:

        n_splits = crowdastro_h5['/wise/cdfs/test_sets'].shape[0]
        n_examples, n_params = training_h5['features'].shape
        n_params += 1  # Bias term.
        methods = ['LR(Norris)', 'LR(Fan)', 'RF(Norris)', 'RF(Fan)']
        model = '{} sklearn.linear_model.LogisticRegression'.format(
                sklearn.__version__)  # No model for RF.

        results = Results(results_h5_path, methods, n_splits, n_examples,
                          n_params, model)

        features = collections.defaultdict(
                lambda: training_h5['features'].value)
        targets = {
            'LR(Norris)': crowdastro_h5['/wise/cdfs/norris_labels'],
            'LR(Fan)': crowdastro_h5['/wise/cdfs/fan_labels'],
            'RF(Norris)': crowdastro_h5['/wise/cdfs/norris_labels'],
            'RF(Fan)': crowdastro_h5['/wise/cdfs/fan_labels'],
        }

        for split_id, test_set in enumerate(
                    crowdastro_h5['/wise/cdfs/test_sets']):
            logging.info('Test {}/{}'.format(split_id + 1, n_splits))
            for method_id, method in enumerate(methods):
                if method.startswith('LR'):
                    runner = runners.lr
                else:
                    runner = runners.rf

                logging.info('Method {} ({}/{})'.format(method, method_id + 1,
                                                        len(methods)))
                runner(results, method, split_id, features[method],
                       targets[method], list(test_set), overwrite=overwrite)

        if plot:
            matplotlib.rcParams['font.family'] = 'serif'
            matplotlib.rcParams['font.serif'] = ['Palatino Linotype']
            vertical_scatter_ba(results,
                    crowdastro_h5['/wise/cdfs/norris_labels'].value)
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crowdastro', default='data/crowdastro.h5',
                        help='HDF5 crowdastro data file')
    parser.add_argument('--training', default='data/training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--results', default='data/results_lr_rf.h5',
                        help='HDF5 results data file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--plot', action='store_true', help='Generate a plot')
    args = parser.parse_args()

    logging.root.setLevel(logging.INFO)

    main(args.crowdastro, args.training, args.results, overwrite=args.overwrite,
         plot=args.plot)
