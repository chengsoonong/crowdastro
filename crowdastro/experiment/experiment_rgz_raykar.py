"""Runs the Raykar et al. (2010) algorithm on the galaxy classification task.

Matthew Alger
The Australian National University
2016
"""

import argparse
import logging
import time

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


def _top_n_accurate_targets(crowdastro_h5, y_true, n_annotators=5,
                            threshold=700):
    """Get the labels of the top n most accurate annotators assessed against
    y_true, above a threshold number of annotations."""
    labels = crowdastro_h5['/wise/cdfs/rgz_raw_labels'].value
    labels_mask = crowdastro_h5['/wise/cdfs/rgz_raw_labels_mask'].value
    labels = numpy.ma.MaskedArray(labels, mask=labels_mask)
    # Compare each annotator to the Norris labels. Get their balanced accuracy.
    annotator_accuracies = []
    for t in range(labels.shape[0]):
        cm = sklearn.metrics.confusion_matrix(y_true[~labels[t].mask],
                                              labels[t][~labels[t].mask])
        if cm.shape[0] == 1:
            continue

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


def top_n_accurate_targets(crowdastro_h5, n_annotators=5, threshold=700):
    """Get the labels of the top n most accurate annotators, assessed
    against the groundtruth, above a threshold number of annotations.
    """
    norris = crowdastro_h5['/wise/cdfs/norris_labels'].value
    return _top_n_accurate_targets(crowdastro_h5, norris,
                                   n_annotators=n_annotators,
                                   threshold=threshold)


def top_n_mv_accurate_targets(crowdastro_h5, n_annotators=5, threshold=700):
    """Get the labels of the top n most accurate annotators, assessed
    against the majority vote, above a threshold number of annotations.
    """
    labels = crowdastro_h5['/wise/cdfs/rgz_raw_labels'].value
    labels_mask = crowdastro_h5['/wise/cdfs/rgz_raw_labels_mask'].value
    labels = numpy.ma.MaskedArray(labels, mask=labels_mask)
    mv = majority_vote(labels)
    return _top_n_accurate_targets(crowdastro_h5, mv,
                                   n_annotators=n_annotators,
                                   threshold=threshold)


def top_n_prolific_targets(crowdastro_h5, n_annotators=5):
    """Get the labels of the top n most prolific annotators."""
    labels = crowdastro_h5['/wise/cdfs/rgz_raw_labels'].value
    labels_mask = crowdastro_h5['/wise/cdfs/rgz_raw_labels_mask'].value
    labels = numpy.ma.MaskedArray(labels, mask=labels_mask)
    n_seen = labels_mask.sum(axis=1)
    top_seen = numpy.argsort(n_seen)[-n_annotators:]
    return labels[top_seen]


def main(crowdastro_h5_path, training_h5_path, results_h5_path,
         overwrite=False, plot=False, n_annotators=10):
    with h5py.File(crowdastro_h5_path, 'r') as crowdastro_h5, \
            h5py.File(training_h5_path, 'r') as training_h5:

        n_splits = crowdastro_h5['/wise/cdfs/test_sets'].shape[0]
        n_examples, n_params = training_h5['features'].shape
        n_params += 1  # Bias term.
        n_params += 1  # Number of annotators.
        n_params += n_annotators * 2  # Alpha and beta.
        methods = [
            'Raykar(Top-{}-prolific)'.format(n_annotators),
            'LR(Top-{}-prolific-MV)'.format(n_annotators),
            'Raykar(Top-{}-accurate)'.format(n_annotators),
            'LR(Top-{}-accurate-MV)'.format(n_annotators),
            'Raykar(Top-{}-est-accurate)'.format(n_annotators),
            'LR(Top-{}-est-accurate-MV)'.format(n_annotators),
        ]
        model = ('{} crowdastro.crowd.raykar.RaykarClassifier, '.format(
                    __version__) +
                 '{} sklearn.linear_model.LogisticRegression, '.format(
                    sklearn.__version__) +
                 '{} crowdastro.crowd.raykar.RaykarClassifier, '.format(
                    __version__) +
                 '{} sklearn.linear_model.LogisticRegression'.format(
                    sklearn.__version__) +
                 '{} crowdastro.crowd.raykar.RaykarClassifier, '.format(
                    __version__) +
                 '{} sklearn.linear_model.LogisticRegression'.format(
                    sklearn.__version__))

        results = Results(results_h5_path, methods, n_splits, n_examples,
                          n_params, model)

        all_features = training_h5['features'].value
        targets = {
            'Raykar(Top-{}-prolific)'.format(n_annotators):
                top_n_prolific_targets(
                    crowdastro_h5,
                    n_annotators=n_annotators),
            'LR(Top-{}-prolific-MV)'.format(n_annotators):
                majority_vote(
                    top_n_prolific_targets(
                        crowdastro_h5,
                        n_annotators=n_annotators)),
            'Raykar(Top-{}-accurate)'.format(n_annotators):
                top_n_accurate_targets(
                    crowdastro_h5,
                    n_annotators=n_annotators),
            'LR(Top-{}-accurate-MV)'.format(n_annotators):
                majority_vote(
                    top_n_accurate_targets(
                        crowdastro_h5,
                        n_annotators=n_annotators)),
            'Raykar(Top-{}-est-accurate)'.format(n_annotators):
                top_n_mv_accurate_targets(
                    crowdastro_h5,
                    n_annotators=n_annotators),
            'LR(Top-{}-est-accurate-MV)'.format(n_annotators):
                majority_vote(
                    top_n_mv_accurate_targets(
                        crowdastro_h5,
                        n_annotators=n_annotators)),
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
                    then = time.time()
                    runner(results, method, split_id, features,
                           targets[method], list(test_set),
                           overwrite=overwrite, n_restarts=1)
                    now = time.time()
                    logging.info('Raykar took {} s'.format(now - then))
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
            vertical_scatter_ba(
                    results,
                    crowdastro_h5['/wise/cdfs/norris_labels'].value,
                    rotation=45, x_tick_offset=-0.5)
            # Add a little space for the labels.
            plt.subplots_adjust(bottom=0.3)
            matplotlib.rcParams['figure.figsize'] = [8.0, 8.0]
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
    parser.add_argument('--annotators', default=10,
                        help='number of annotators', type=int)
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
         plot=args.plot, n_annotators=args.annotators)
