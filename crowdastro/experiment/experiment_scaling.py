"""Compares logistic regression to random forests, trained on Norris and Fan.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy
import sklearn
import sklearn.decomposition
import sklearn.preprocessing

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
        methods = ['Scaled', 'Unscaled', 'Normalised', 'Normal+scale',
                   'Whitened', 'Scale+normal', 'Normal+scale+white',
                   'Normal astro', 'Scale astro', 'Normal+scale astro']
        model = '{} sklearn.linear_model.LogisticRegression'.format(
                sklearn.__version__)

        results = Results(results_h5_path, methods, n_splits, n_examples,
                          n_params, model)

        pca = sklearn.decomposition.PCA(whiten=True)
        whitened_features = pca.fit_transform(training_h5['features'])

        pca = sklearn.decomposition.PCA(whiten=True)
        nsw_features = pca.fit_transform(sklearn.preprocessing.normalize(
            sklearn.preprocessing.scale(training_h5['features'])))

        features = {
            'Scaled': sklearn.preprocessing.scale(training_h5['features']),
            'Unscaled': training_h5['features'],
            'Normalised': sklearn.preprocessing.normalize(
                training_h5['features']),
            'Normal+scale': sklearn.preprocessing.normalize(
                sklearn.preprocessing.scale(training_h5['features'])),
            'Whitened': whitened_features,
            'Scale+normal': sklearn.preprocessing.scale(
                sklearn.preprocessing.normalize(training_h5['features'])),
            'Normal+scale+white': nsw_features,
            'Normal astro': numpy.hstack([
                sklearn.preprocessing.normalize(training_h5['features'][:, :8]),
                training_h5['features'][:, 8:],
            ]),
            'Scale astro': numpy.hstack([
                sklearn.preprocessing.scale(training_h5['features'][:, :8]),
                training_h5['features'][:, 8:],
            ]),
            'Normal+scale astro': numpy.hstack([
                sklearn.preprocessing.normalize(
                    sklearn.preprocessing.scale(
                        training_h5['features'][:, :8])),
                training_h5['features'][:, 8:],
            ]),
        }
        targets = crowdastro_h5['/wise/cdfs/norris_labels']

        for split_id, test_set in enumerate(
                    crowdastro_h5['/wise/cdfs/test_sets']):
            logging.info('Test {}/{}'.format(split_id + 1, n_splits))
            for method_id, method in enumerate(methods):
                logging.info('Method {} ({}/{})'.format(method, method_id + 1,
                                                        len(methods)))
                runners.lr(results, method, split_id, features[method],
                           targets, list(test_set), overwrite=overwrite)

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
    parser.add_argument('--results', default='data/results_scaling.h5',
                        help='HDF5 results data file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--plot', action='store_true', help='Generate a plot')
    args = parser.parse_args()

    logging.root.setLevel(logging.INFO)

    main(args.crowdastro, args.training, args.results, overwrite=args.overwrite,
         plot=args.plot)
