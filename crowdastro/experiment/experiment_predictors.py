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
import sklearn.metrics

from .. import __version__
from . import runners
from .experiment_rgz_raykar import top_n_mv_accurate_targets
from .results import Results
from ..config import config
from ..plot import vertical_scatter_ba


def raw_majority_vote_experiment(results, method, split_id, n_params,
                                 crowdastro_h5):
    # For each galaxy, find a percentage for that galaxy. >= 0.5 will identify a
    # galaxy as containing an AGN.
    ir_survey = crowdastro_h5.attrs['ir_survey']
    labels = crowdastro_h5['/{}/cdfs/rgz_raw_labels'.format(ir_survey)].value
    labels_mask = crowdastro_h5['/{}/cdfs/rgz_raw_labels_mask'.format(
        ir_survey)].value
    labels = numpy.ma.MaskedArray(labels, mask=labels_mask)
    percentages = labels.mean(axis=0)
    results.store_trial(method, split_id, percentages, numpy.zeros((n_params,)))


def main(crowdastro_h5_path, training_h5_path, results_h5_path,
         overwrite=False, plot=False, n_annotators=50):
    with h5py.File(crowdastro_h5_path, 'r') as crowdastro_h5, \
            h5py.File(training_h5_path, 'r') as training_h5:
        ir_survey = training_h5.attrs['ir_survey']
        ir_survey_ = crowdastro_h5.attrs['ir_survey']
        assert ir_survey == ir_survey_

        n_splits = crowdastro_h5['/{}/cdfs/test_sets'.format(
            ir_survey)].shape[0]
        n_examples, n_params = training_h5['features'].shape
        n_params += 1  # Bias term.
        n_params += 2 * n_annotators  # Raykar annotator model.
        n_params += 1  # Metadata.
        methods = ['LR(Norris)', 'LR(Fan)', 'LR(RGZ-MV)',
                   'Raykar(RGZ-Top-{})'.format(n_annotators),
                   'RGZ-Raw-MV']
        model = '{} sklearn.linear_model.LogisticRegression,'.format(
                sklearn.__version__) * 3 + \
            '{} crowdastro.crowd.raykar.RaykarClassifier'.format(
                __version__)

        results = Results(results_h5_path, methods, n_splits, n_examples,
                          n_params, model)

        features = collections.defaultdict(
                lambda: training_h5['features'].value)
        targets = {
            'LR(Norris)': crowdastro_h5['/{}/cdfs/norris_labels'.format(
                ir_survey)],
            'LR(Fan)': crowdastro_h5['/{}/cdfs/fan_labels'.format(ir_survey)],
            'LR(RGZ-MV)': training_h5['labels'],
            'Raykar(RGZ-Top-{})'.format(n_annotators):
                top_n_mv_accurate_targets(
                    crowdastro_h5,
                    n_annotators=n_annotators),
        }

        for split_id, test_set in enumerate(
                    crowdastro_h5['/{}/cdfs/test_sets'.format(ir_survey)]):
            logging.info('Test {}/{}'.format(split_id + 1, n_splits))
            for method_id, method in enumerate(methods):
                logging.info('Method {} ({}/{})'.format(method, method_id + 1,
                                                        len(methods)))

                if ir_survey == 'swire':
                    features_ = numpy.nan_to_num(features[method])
                    p2, p98 = numpy.percentile(
                        features_[:config['surveys']['swire']['n_features']],
                        [2, 98])
                    features_[features_ > p98] = p98
                    features_[features_ < 2] = p2
                    logging.debug('Clamping to range {} -- {}'.format(p2, p98))
                else:
                    features_ = features[method]

                if method.startswith('LR'):
                    runners.lr(results, method, split_id, features_,
                               targets[method], list(test_set),
                               overwrite=overwrite)
                elif method.startswith('Raykar'):
                    runners.raykar(results, method, split_id, features_,
                                   targets[method], list(test_set),
                                   overwrite=overwrite, n_restarts=5,
                                   downsample=True)
                elif method == 'RGZ-Raw-MV':
                    raw_majority_vote_experiment(
                        results, method, split_id,
                        n_params, crowdastro_h5)

        if plot:
            matplotlib.rcParams['font.family'] = 'serif'
            matplotlib.rcParams['font.serif'] = ['Palatino Linotype']
            plt.figure(figsize=(6, 4))
            vertical_scatter_ba(
                results,
                crowdastro_h5['/{}/cdfs/norris_labels'.format(ir_survey)].value,
                violin=True, rotation=45, x_tick_offset=-0.5)
            plt.subplots_adjust(bottom=0.33)
            plt.show()

        for method in methods:
            probs = results[method].mean(axis=0)
            assert probs.shape == targets['LR(Norris)'].shape
            cm = sklearn.metrics.confusion_matrix(
                targets['LR(Norris)'],
                probs.round())
            cm /= cm.sum(axis=1)
            print(method, '\n', cm)


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
