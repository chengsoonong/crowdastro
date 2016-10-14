"""Query by committee on the Radio Galaxy Zoo MV dataset.

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
import sklearn.cross_validation
import sklearn.linear_model

from crowdastro.crowd.util import balanced_accuracy
from crowdastro.active_learning import random_sampler, qbc_sampler
from crowdastro.plot import fillbetween


def main(crowdastro_h5_path, training_h5_path, results_npy_path,
         overwrite=False, plot=False, n_trials=25):
    with h5py.File(crowdastro_h5_path, 'r') as crowdastro_h5:
        with h5py.File(training_h5_path, 'r') as training_h5:
            n_instances, = training_h5['labels'].shape
            n_splits, n_test_instances = crowdastro_h5[
                '/wise/cdfs/test_sets'].shape
            n_train_indices = n_instances - n_test_instances
            n_methods = 2
            instance_counts = [int(i) for i in numpy.logspace(
                numpy.log10(10), numpy.log10(n_train_indices), n_trials)]
            results = numpy.zeros((n_methods, n_splits, n_trials))
            features = training_h5['features'].value
            labels = training_h5['labels'].value
            norris = crowdastro_h5['/wise/cdfs/norris_labels'].value

            for method_index, Sampler in enumerate([
                    qbc_sampler.QBCSampler, random_sampler.RandomSampler]):
                logging.debug(str(Sampler))
                for split_id, split in enumerate(
                        crowdastro_h5['/wise/cdfs/test_sets'].value):
                    logging.info('Running split {}/{}'.format(split_id + 1,
                                                              n_splits))
                    # Set of indices that are not the testing set. This is where
                    # it's valid to query from.
                    train_indices = set(numpy.arange(n_instances))
                    for i in split:
                        train_indices.remove(i)
                    train_indices = sorted(train_indices)

                    # The masked set of labels we can query.
                    queryable_labels = numpy.ma.MaskedArray(
                        training_h5['labels'][train_indices],
                        mask=numpy.ones(n_train_indices))

                    # Initialise by selecting instance_counts[0] random labels,
                    # stratified.
                    init_indices, _ = sklearn.cross_validation.train_test_split(
                        numpy.arange(n_train_indices),
                        train_size=instance_counts[0],
                        stratify=queryable_labels.data)
                    queryable_labels.mask[init_indices] = 0
                    sampler = Sampler(
                        features[train_indices], queryable_labels,
                        sklearn.linear_model.LogisticRegression,
                        classifier_params={'class_weight': 'balanced'})

                    results[method_index, split_id, 0] = sampler.ba(
                        features[split], norris[split])
                    for count_index, count in enumerate(instance_counts[1:]):
                        # Query until we have seen count labels.
                        n_required_queries = count - (~sampler.labels.mask).sum()
                        assert n_required_queries >= 0
                        # Make that many queries.
                        logging.debug('Making {} queries.'.format(
                            n_required_queries))
                        queries = sampler.sample_indices(n_required_queries)
                        queried_labels = queryable_labels.data[queries]
                        sampler.add_labels(queries, queried_labels)
                        logging.debug('Total labels known: {}'.format(
                            (~sampler.labels.mask).sum()))
                        results[
                            method_index, split_id, count_index + 1
                        ] = sampler.ba(features[split], norris[split])

                    # with open(results_npy_path, 'w') as f:
                    #     numpy.save(f, results, allow_pickle=False)

                    # TODO(MatthewJA): Implement overwrite parameter.

            fillbetween(instance_counts, list(zip(*results[0, :])))
            fillbetween(instance_counts, list(zip(*results[1, :])),
                        facecolour='blue', edgecolour='blue', facealpha=0.2)
            plt.legend(['QBC', 'Passive'])
            plt.xscale('log')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crowdastro', default='data/crowdastro.h5',
                        help='HDF5 crowdastro data file')
    parser.add_argument('--training', default='data/training.h5',
                        help='HDF5 training data file')
    parser.add_argument('--results', default='data/results_rgz_qbc.npy',
                        help='NPY results data file')
    # parser.add_argument('--overwrite', action='store_true',
    #                     help='Overwrite existing results')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--plot', action='store_true', help='Generate a plot')
    args = parser.parse_args()

    if args.verbose:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    main(args.crowdastro, args.training, args.results, overwrite=True,
         plot=args.plot)
