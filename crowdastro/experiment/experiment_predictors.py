"""Runs logistic regression on multiple label sets and test set splits.

- Radio Galaxy Zoo majority vote labels
- Norris et al. (2006) labels
- Fan et al. (2015) labels

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
        methods = ['LR(Norris)', 'LR(Fan)', 'LR(RGZ-MV)', 'LR(RGZ-Raw)']
        model = '{} sklearn.linear_model.LogisticRegression'.format(
                sklearn.__version__)

        results = Results(results_h5_path, methods, n_splits, n_examples,
                          n_params, model)

        features = collections.defaultdict(
                lambda: training_h5['features'].value)
        targets = {
            'LR(Norris)': crowdastro_h5['/wise/cdfs/norris_labels'],
            'LR(Fan)': crowdastro_h5['/wise/cdfs/fan_labels'],
            'LR(RGZ-MV)': training_h5['labels'],
            'LR(RGZ-Raw)': None,
        }

        # Build raw features/labels.
        rgz_raw_labels = crowdastro_h5['/wise/cdfs/rgz_raw_labels']
        rgz_raw_labels_mask = crowdastro_h5['/wise/cdfs/rgz_raw_labels_mask']
        raw_features = []
        raw_labels = []
        for anno_labels, anno_mask in zip(rgz_raw_labels, rgz_raw_labels_mask):
            seen = (~anno_mask).nonzero()[0]
            raw_features_ = training_h5['features'][seen, :]
            raw_labels_ = anno_labels[seen]
            raw_features.extend(raw_features_)
            raw_labels.extend(raw_labels_)
        raw_features = numpy.array(raw_features)
        raw_labels = numpy.array(raw_labels)
        assert raw_features.shape[0] == raw_labels.shape[0]
        print(raw_features.shape)
        features['LR(RGZ-Raw)'] = raw_features
        targets['LR(RGZ-Raw)'] = raw_labels

        for split_id, test_set in enumerate(
                    crowdastro_h5['/wise/cdfs/test_sets']):
            logging.info('Test {}/{}'.format(split_id + 1, n_splits))
            for method_id, method in enumerate(methods):
                logging.info('Method {} ({}/{})'.format(method, method_id + 1,
                                                        len(methods)))
                runners.lr(results, method, split_id, features[method],
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
    parser.add_argument('--results', default='data/results.h5',
                        help='HDF5 results data file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--plot', action='store_true', help='Generate a plot')
    args = parser.parse_args()

    logging.root.setLevel(logging.INFO)

    main(args.crowdastro, args.training, args.results, overwrite=args.overwrite,
         plot=args.plot)
