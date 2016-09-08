"""Runs logistic regression on the Norris et al. (2010) labels with different
numbers of labels.

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
from ..config import config
from ..plot import vertical_scatter_ba

ARCMIN = 1 / 60  # deg
ATLAS_IMAGE_SIZE = (config['surveys']['atlas']['fits_width'] * 
                    config['surveys']['atlas']['fits_height'])


def main(crowdastro_h5_path, training_h5_path, results_h5_path,
         overwrite=False, plot=False):
    with h5py.File(crowdastro_h5_path, 'r') as crowdastro_h5, \
            h5py.File(training_h5_path, 'r') as training_h5:

        n_splits = 30
        n_examples, n_params = training_h5['features'].shape
        n_params += 1  # Bias term.

        methods = [str(i) for i in [10, 30, 50, 100, 300, 500, 1000, 1500,
                                    2000]]
        model = '{} sklearn.linear_model.LogisticRegression'.format(
                sklearn.__version__)

        results = Results(results_h5_path, methods, n_splits, n_examples,
                          n_params, model)

        features = training_h5['features'].value
        targets = crowdastro_h5['/wise/cdfs/norris_labels'].value.astype(bool)

        for method in methods:
            method = int(method)
            logging.info('Testing {} ATLAS objects'.format(method))
            for split in range(n_splits):
                logging.info('Test {}/{}'.format(split + 1, n_splits))
                # Sample random ATLAS objects.
                atlas_ids = numpy.arange(
                        crowdastro_h5['/atlas/cdfs/numeric'].shape[0])
                numpy.random.shuffle(atlas_ids)
                train_atlas = atlas_ids[:method]
                # Get nearby IR objects.
                nearby = set()
                for atlas_id in train_atlas:
                    nearby_me = (crowdastro_h5['/atlas/cdfs/numeric'][
                        atlas_id, 2 + ATLAS_IMAGE_SIZE:] <= ARCMIN).nonzero()[0]
                    for i in nearby_me:
                        nearby.add(i)
                train_ir = nearby
                # This is the training set. Now, get everything else.
                ir_ids = range(crowdastro_h5['/wise/cdfs/numeric'].shape[0])
                test_ir = set(ir_ids) - train_ir
                train_ir = sorted(train_ir)
                test_ir = sorted(test_ir)
                runners.lr(results, str(method), split, features,
                           targets, test_indices=test_ir,
                           overwrite=overwrite)

        if plot:
            matplotlib.rcParams['font.family'] = 'serif'
            matplotlib.rcParams['font.serif'] = ['Palatino Linotype']
            vertical_scatter_ba(results,
                    crowdastro_h5['/wise/cdfs/norris_labels'].value,
                    violin=True)
            plt.xlabel('Number of ATLAS training objects')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crowdastro', default='data/crowdastro.h5',
                        help='HDF5 crowdastro data file')
    parser.add_argument('--training', default='data/training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--results', default='data/results_learning_curve.h5',
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

    main(args.crowdastro, args.training, args.results, overwrite=args.overwrite,
         plot=args.plot)
