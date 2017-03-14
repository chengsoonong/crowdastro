"""Runs logistic regression on multiple label sets and test set splits.

- Radio Galaxy Zoo majority vote labels
- Norris et al. (2006) labels

Compares the results to the majority vote from the Radio Galaxy Zoo.

Matthew Alger
The Australian National University
2017
"""

import argparse
import logging

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import sklearn
import sklearn.metrics

from . import runners
from ..crowd.util import balanced_accuracy
from .results import Results
from ..plot import vertical_scatter_ba

def raw_majority_vote_experiment(results, method, split_id, n_params,
                                 crowdastro_h5, test_set):
    # For each galaxy, find a percentage for that galaxy. >= 0.5 will identify a
    # galaxy as containing an AGN.
    ir_survey = crowdastro_h5.attrs['ir_survey']
    ir_prefix = '/{}/cdfs/'.format(ir_survey)
    labels = crowdastro_h5[ir_prefix + 'rgz_raw_labels'].value[:, test_set]
    labels_mask = crowdastro_h5[ir_prefix + 'rgz_raw_labels_mask'].value[:, test_set]
    labels = numpy.ma.MaskedArray(labels, mask=labels_mask)
    percentages = labels.mean(axis=0)
    print('Average percentage is {} for test set {}'.format(percentages.mean(), test_set[:100]))
    results.store_trial(method, split_id, percentages, numpy.zeros((n_params,)),
                        indices=list(test_set))


def norris_experiment(results, method, split_id, n_params, crowdastro_h5):
    ir_survey = crowdastro_h5.attrs['ir_survey']
    ir_prefix = '/{}/cdfs/'.format(ir_survey)
    labels = crowdastro_h5[ir_prefix + 'norris_labels'].value
    results.store_trial(method, split_id, labels, numpy.zeros((n_params,)))


def main(crowdastro_h5_path, training_h5_path, results_h5_path,
         overwrite=False, plot=False, n_annotators=50):
    with h5py.File(crowdastro_h5_path, 'r') as crowdastro_h5, \
            h5py.File(training_h5_path, 'r') as training_h5:
        # Input validation.
        ir_survey = training_h5.attrs['ir_survey']
        ir_survey_ = crowdastro_h5.attrs['ir_survey']
        assert ir_survey == ir_survey_

        # We are using WISE for the paper, so ensure that we are also using WISE
        # for the results...
        assert ir_survey == 'wise'

        ir_prefix = '/{}/cdfs/'.format(ir_survey)

        # Store some metadata:
        # Number of test sets
        n_splits = crowdastro_h5[ir_prefix + 'test_sets'].shape[0]
        # Number of training + testing instances
        n_examples, n_params = training_h5['features'].shape
        # Number of parameters to store a serialised model
        n_params += 1  # Bias term for LR.
        n_params += 1  # Metadata for LR.
        # Model used for classification.
        model = '{} sklearn.linear_model.LogisticRegression,'.format(
                sklearn.__version__)

        # Classification methods (predictors) we will report.
        # LR(Dataset) represents logistic regression trained on the dataset.
        methods = [
            'Norris',
            'LR(Norris)',
            'RGZ MV',
            'LR(RGZ MV)',
        ]

        # Set up a results object to store our results.
        results = Results(results_h5_path, methods, n_splits, n_examples,
                          n_params, model)

        # Features are the same for all predictors.
        features = training_h5['features'].value
        # Targets differ depending on the training set.
        targets = {
            'LR(Norris)': crowdastro_h5[ir_prefix + 'norris_labels'],
            'LR(RGZ MV)': training_h5['labels'],
        }

        # Conduct the experiment.
        # For each test set...
        for split_id, test_set in enumerate(
                    crowdastro_h5['/{}/cdfs/test_sets'.format(ir_survey)]):
            logging.info('Test {}/{}'.format(split_id + 1, n_splits))
            # For each predictor...
            for method_id, method in enumerate(methods):
                logging.info('Method {} ({}/{})'.format(method, method_id + 1,
                                                        len(methods)))

                if method.startswith('LR'):
                    # Run logistic regression.
                    runners.lr(results, method, split_id, features,
                               targets[method], list(test_set),
                               overwrite=overwrite)
                elif method == 'RGZ MV':
                    # Compute the majority vote on the test set.
                    raw_majority_vote_experiment(results, method, split_id,
                                                 n_params, crowdastro_h5,
                                                 test_set)
                elif method == 'Norris':
                    norris_experiment(results, method, split_id, n_params,
				      crowdastro_h5)
        if plot:
            import time
            matplotlib.rcParams['font.family'] = 'serif'
            matplotlib.rcParams['font.serif'] = ['Palatino Linotype']
            plt.figure(figsize=(6, 6))
            vertical_scatter_ba(
                results,
                crowdastro_h5[ir_prefix + 'norris_labels'].value,
                violin=False, rotation='vertical')
            plt.subplots_adjust(bottom=0.28)
            plt.savefig('plot{}.pdf'.format(time.time()))

        # Compute the mean and stdev for each predictor.
        for method in methods:
            total_cm = numpy.zeros((2, 2, n_splits))
            accuracies = []
            for split in range(n_splits):
                probs = results[method, split][results.get_mask(method, split)]
                assert probs.shape == targets[
                    'LR(Norris)'][results.get_mask(method, split)].shape
                total_cm[:, :, split] = sklearn.metrics.confusion_matrix(
                    targets['LR(Norris)'][results.get_mask(method, split)],
                    probs.round()).astype(float)
                accuracies.append(balanced_accuracy(
                    targets['LR(Norris)'][results.get_mask(method, split)],
                    probs.round()))
            # Normalise confusion matrices.
            total_cm /= total_cm.sum(axis=1).reshape((-1, 1, n_splits))
            # Compute mean and stdev.
            mean_cm = total_cm.mean(axis=2)
            std_cm = total_cm.std(axis=2)
            mean_ba = numpy.mean(accuracies)
            std_ba = numpy.std(accuracies)
            print(method, '(mean) \n', mean_cm, '& ba:', mean_ba)
            print(method, '(std) \n', std_cm, '& ba:', std_ba)


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
