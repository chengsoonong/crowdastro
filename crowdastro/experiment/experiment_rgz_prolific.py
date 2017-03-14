"""Tests accuracies of RGZ volunteers against Norris.

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

from ..crowd.util import balanced_accuracy
from ..plot import vertical_scatter


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

        ir_survey = crowdastro_h5.attrs['ir_survey']

        # Conduct the experiment.
        # For each test set...
        bas = []
        for split_id, test_set in enumerate(
                    crowdastro_h5['/{}/cdfs/test_sets'.format(ir_survey)]):
            logging.info('Test {}/{}'.format(split_id + 1, n_splits))
            ir_prefix = '/{}/cdfs/'.format(ir_survey)
            labels = crowdastro_h5[ir_prefix + 'rgz_raw_labels'].value
            labels_mask = crowdastro_h5[ir_prefix + 'rgz_raw_labels_mask']

            for labeller_idx, labeller_labels in enumerate(labels):
                labeller_mask = labels_mask[labeller_idx, :]
                labeller_labels = numpy.ma.MaskedArray(
                    labeller_labels, mask=labeller_mask)
                # Compare to Norris.
                norris_labels = crowdastro_h5['{}/cdfs/norris_labels'].format(
                    ir_survey)[labeller_mask]
                ba = balanced_accuracy(norris_labels, labeller_labels)
                bas.append(ba)

        if plot:
            import time
            matplotlib.rcParams['font.family'] = 'serif'
            matplotlib.rcParams['font.serif'] = ['Palatino Linotype']
            plt.figure(figsize=(6, 6))
            vertical_scatter(
                ['RGZ Volunteers'],
                [bas],
                violin=False)
            plt.savefig('plot{}.pdf'.format(time.time()))

        print('BA: ({} +- {})%'.format(numpy.mean(bas), numpy.std(bas)))


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
