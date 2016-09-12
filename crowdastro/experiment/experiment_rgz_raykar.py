"""Runs the Raykar et al. (2010) algorithm on the galaxy classification task.

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
from .results import Results
from .. import __version__
from ..plot import vertical_scatter_ba


def top_n_targets(crowdastro_h5, n_annotators=5, threshold=700):
    # Compute the majority vote.
    labels = crowdastro_h5['/wise/cdfs/rgz_raw_labels'].value
    labels_mask = crowdastro_h5['/wise/cdfs/rgz_raw_labels_mask'].value
    labels = numpy.ma.MaskedArray(labels, mask=labels_mask)
    mv = labels.mean(axis=0).round()
    # Compare each annotator to the majority vote. Get their balanced accuracy.
    annotator_accuracies = []
    for t in range(labels.shape[0]):
        cm = sklearn.metrics.confusion_matrix(mv[~labels[t].mask],
                                              labels[t][~labels[t].mask])
        tp = cm[1, 1]
        n, p = cm.sum(axis=1)
        tn = cm[0, 0]
        if not n or not p or p + n < threshold:
            annotator_accuracies.append(0)
            continue

        ba = (tp / p + tn / n) / 2
        annotator_accuracies.append(ba)
    ranked_annotators = numpy.argsort(annotator_accuracies)
    top_n_annotators = ranked_annotators[-n_annotators:]

    top_labels = []
    top_labels_mask = []
    for t in top_n_annotators:
        top_labels.append(labels[t])
        top_labels_mask.append(labels_mask[t])
    return numpy.ma.MaskedArray(top_labels, mask=top_labels_mask)


def main(crowdastro_h5_path, training_h5_path, results_h5_path,
         overwrite=False, plot=False):
    with h5py.File(crowdastro_h5_path, 'r') as crowdastro_h5, \
            h5py.File(training_h5_path, 'r') as training_h5:

        n_splits = crowdastro_h5['/wise/cdfs/test_sets'].shape[0]
        n_examples, n_params = training_h5['features'].shape
        n_params += 1  # Bias term.
        n_params += crowdastro_h5['/wise/cdfs/rgz_raw_labels'].shape[0] * 2
        methods = ['Top-5-annotators', 'Top-10-annotators', 'PCA-10D']
        model = '{} crowdastro.crowd.raykar.RaykarClassifier'.format(
                __version__)

        results = Results(results_h5_path, methods, n_splits, n_examples,
                          n_params, model)

        all_features = training_h5['features'].value
        targets = {
            'Top-5-annotators': top_n_targets(crowdastro_h5, n_annotators=5),
            'Top-10-annotators': top_n_targets(crowdastro_h5, n_annotators=10),
            'PCA-10D': numpy.ma.MaskedArray(
                    crowdastro_h5['/wise/cdfs/rgz_raw_labels'].value,
                    mask=crowdastro_h5['/wise/cdfs/rgz_raw_labels_mask'].value,
            ),
        }

        for split_id, test_set in enumerate(
                    crowdastro_h5['/wise/cdfs/test_sets']):
            logging.info('Test {}/{}'.format(split_id + 1, n_splits))
            for method_id, method in enumerate(methods):
                logging.info('Method {} ({}/{})'.format(method, method_id + 1,
                                                        len(methods)))

                if method == 'PCA-10D':
                    pca = sklearn.decomposition.PCA(n_components=10)
                    train_set = range(
                            crowdastro_h5['/wise/cdfs/numeric'].shape[0])
                    train_set = sorted(set(train_set) - set(test_set))
                    pca.fit(all_features[train_set])
                    features = pca.transform(all_features)
                else:
                    features = all_features

                runners.raykar(results, method, split_id, features,
                               targets[method], list(test_set),
                               overwrite=overwrite, n_restarts=1)

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
    parser.add_argument('--results', default='data/results_rgz_raykar.h5',
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
