"""Tests the influence of features by finding performance when they are removed.

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

from crowdastro.config import config
from crowdastro.experiment import runners
from crowdastro.experiment.results import Results
from crowdastro.plot import vertical_scatter_ba


def main(crowdastro_h5_path, training_h5_path, results_h5_path,
         overwrite=False, plot=False):
    with h5py.File(crowdastro_h5_path, 'r') as crowdastro_h5:
        with h5py.File(training_h5_path, 'r') as training_h5:
            assert training_h5.attrs['ir_survey'] == 'wise'
            # TODO(MatthewJA): Add SWIRE compatibility.

            n_splits = crowdastro_h5['/wise/cdfs/test_sets'].shape[0]
            n_examples, n_params = training_h5['features'].shape
            n_params += 1  # Bias term.

            all_features = training_h5['features']
            n_astro_features = config['surveys']['wise']['n_features']
            distance_i = n_astro_features - 1

            features = {
                'All features': all_features[:],
                'CNN only': all_features[:, n_astro_features:],
                'Magnitudes only': all_features[:, :distance_i],
                'Distance only': all_features[:, distance_i:distance_i + 1],
                'No CNN': all_features[:, :n_astro_features],
                'No magnitudes': all_features[:, 6:],
                'No distance': numpy.hstack(
                    [all_features[:, :6], all_features[:, 7:]]),
                'No CNN + no $w1$': all_features[:, 1:n_astro_features],
                'No CNN + no $w2$': numpy.hstack([
                    all_features[:, :1],
                    all_features[:, 2:n_astro_features]]),
                'No CNN + no $w3$': numpy.hstack([
                    all_features[:, :2],
                    all_features[:, 3:n_astro_features]]),
                'No CNN + no $w4$': numpy.hstack([
                    all_features[:, :3],
                    all_features[:, 4:n_astro_features]]),
                'No CNN + no $w1 - w2$': numpy.hstack([
                    all_features[:, :4],
                    all_features[:, 5:n_astro_features]]),
                'No CNN + no $w2 - w3$': numpy.hstack([
                    all_features[:, :5],
                    all_features[:, 6:n_astro_features]]),
            }
            assert n_astro_features == 7

            model = '{} sklearn.linear_model.LogisticRegression'.format(
                    sklearn.__version__)

            results = Results(results_h5_path, sorted(features), n_splits,
                              n_examples, n_params, model)

            for split_id, test_set in enumerate(
                        crowdastro_h5['/wise/cdfs/test_sets']):
                logging.info('Test {}/{}'.format(split_id + 1, n_splits))
                for method_id, method in enumerate(sorted(features)):
                    logging.info('Method {} ({}/{})'.format(
                        method, method_id + 1, len(features)))
                    runners.lr(
                        results,
                        method,
                        split_id,
                        features[method],
                        crowdastro_h5['/wise/cdfs/norris_labels'],
                        list(test_set),
                        overwrite=overwrite)

            if plot:
                matplotlib.rcParams['font.family'] = 'serif'
                matplotlib.rcParams['font.serif'] = ['Palatino Linotype']
                plt.figure(figsize=[11, 6])
                vertical_scatter_ba(
                    results, crowdastro_h5['/wise/cdfs/norris_labels'].value,
                    ylim=(50, 100), violin=True, rotation='vertical',
                    minorticks=False)
                plt.subplots_adjust(bottom=0.3)
                plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crowdastro', default='data/crowdastro.h5',
                        help='HDF5 crowdastro data file')
    parser.add_argument('--training', default='data/training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--results', default='data/results_feature_ablation.h5',
                        help='HDF5 results data file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing results')
    parser.add_argument('--plot', action='store_true', help='Generate a plot')
    args = parser.parse_args()

    logging.root.setLevel(logging.INFO)

    main(args.crowdastro, args.training, args.results, overwrite=args.overwrite,
         plot=args.plot)
