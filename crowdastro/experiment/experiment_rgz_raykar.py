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
from ..crowd.util import majority_vote
from ..crowd.raykar import RaykarClassifier
from ..plot import vertical_scatter_ba


def top_n_accurate_targets(crowdastro_h5, n_annotators=5, threshold=700):
    """Get the labels of the top n most accurate annotators, assessed
    against the groundtruth, above a threshold number of annotations.
    """
    labels = crowdastro_h5['/wise/cdfs/rgz_raw_labels'].value
    labels_mask = crowdastro_h5['/wise/cdfs/rgz_raw_labels_mask'].value
    norris = crowdastro_h5['/wise/cdfs/norris_labels'].value
    labels = numpy.ma.MaskedArray(labels, mask=labels_mask)
    # Compare each annotator to the majority vote. Get their balanced accuracy.
    annotator_accuracies = []
    for t in range(labels.shape[0]):
        cm = sklearn.metrics.confusion_matrix(norris[~labels[t].mask],
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

    return labels[top_n_annotators]


def top_n_prolific_targets(crowdastro_h5, n_annotators=5):
    """Get the labels of the top n most prolific annotators."""
    labels = crowdastro_h5['/wise/cdfs/rgz_raw_labels'].value
    labels_mask = crowdastro_h5['/wise/cdfs/rgz_raw_labels_mask'].value
    labels = numpy.ma.MaskedArray(labels, mask=labels_mask)
    n_seen = labels_mask.sum(axis=1)
    top_seen = numpy.argsort(n_seen)[-n_annotators:]
    return labels[top_seen]


def main(crowdastro_h5_path, training_h5_path, results_h5_path,
         overwrite=False, plot=False):
    with h5py.File(crowdastro_h5_path, 'r') as crowdastro_h5, \
            h5py.File(training_h5_path, 'r') as training_h5:

        n_splits = crowdastro_h5['/wise/cdfs/test_sets'].shape[0]
        n_examples, n_params = training_h5['features'].shape
        n_params += 1  # Bias term.
        n_params += 1  # Number of annotators.
        n_params += crowdastro_h5['/wise/cdfs/rgz_raw_labels'].shape[0] * 2
        methods = [
            'Raykar(Top-10-prolific)',
            'LR(Top-10-prolific-MV)',
            'Raykar(Top-10-accurate)',
            'LR(Top-10-accurate-MV)',
        ]
        model = ('{} crowdastro.crowd.raykar.RaykarClassifier, '.format(
                    __version__) +
                 '{} sklearn.linear_model.LogisticRegression, '.format(
                    sklearn.__version__) +
                 '{} crowdastro.crowd.raykar.RaykarClassifier, '.format(
                    __version__) +
                 '{} sklearn.linear_model.LogisticRegression'.format(
                    sklearn.__version__))

        results = Results(results_h5_path, methods, n_splits, n_examples,
                          n_params, model)

        all_features = training_h5['features'].value
        targets = {
            'Raykar(Top-10-prolific)':
                top_n_prolific_targets(crowdastro_h5, n_annotators=10),
            'LR(Top-10-prolific-MV)':
                majority_vote(
                    top_n_prolific_targets(crowdastro_h5, n_annotators=10)),
            'Raykar(Top-10-accurate)':
                top_n_accurate_targets(crowdastro_h5, n_annotators=10),
            'LR(Top-10-accurate-MV)':
                majority_vote(
                    top_n_accurate_targets(crowdastro_h5, n_annotators=10)),
        }

        for split_id, test_set in enumerate(
                    crowdastro_h5['/wise/cdfs/test_sets']):
            logging.info('Test {}/{}'.format(split_id + 1, n_splits))
            for method_id, method in enumerate(methods):
                logging.info('Method {} ({}/{})'.format(method, method_id + 1,
                                                        len(methods)))
                features = all_features
                if method.startswith('Raykar'):
                    runner = runners.raykar
                    runner(results, method, split_id, features,
                           targets[method], list(test_set),
                           overwrite=overwrite, n_restarts=1)
                    model = results.get_model(method, split_id)
                    rc = RaykarClassifier.unserialise(model)
                    logging.info('{} alpha: {}'.format(method, rc.a_))
                    logging.info('{} beta: {}'.format(method, rc.b_))
                else:
                    runner = runners.lr
                    runner(results, method, split_id, features,
                           targets[method], list(test_set),
                           overwrite=overwrite)

        if plot:
            matplotlib.rcParams['font.family'] = 'serif'
            matplotlib.rcParams['font.serif'] = ['Palatino Linotype']
            # Make the plot a little bit wider for the x labels.
            matplotlib.rcParams['figure.figsize'] = [10.0, 6.0]
            vertical_scatter_ba(
                    results,
                    crowdastro_h5['/wise/cdfs/norris_labels'].value)
            plt.ylim((0, 1))
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
