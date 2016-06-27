"""Tests active learning agents with pool-based binary classification.

Matthew Alger
The Australian National University
2016
"""

import multiprocessing
import multiprocessing.pool

import matplotlib.pyplot as plt
import numpy
import sklearn.cross_validation
import sklearn.datasets
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics

from .uncertainty_sampler import ConfidenceUncertaintySampler
from .random_sampler import RandomSampler
from .qbc_sampler import QBCSampler


def run_tests():
    xs, groundtruth = sklearn.datasets.make_classification(
            n_samples=400, n_features=10, n_informative=5, n_redundant=2,
            n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None,
            flip_y=0.0, class_sep=0.9, hypercube=True, shift=0.0, scale=1.0,
            shuffle=True, random_state=None)

    xs_train, xs_test, groundtruth_train, ts_test = (
            sklearn.cross_validation.train_test_split(xs, groundtruth,
                                                      test_size=0.2))


    init_percentage = 0.50  # Start with 50% of all labels.
    mask = numpy.random.binomial(1, 1 - init_percentage,
                                 size=groundtruth_train.shape)
    known_labels = numpy.ma.masked_array(groundtruth_train, mask=mask)
    assert xs_train.shape[0] == known_labels.shape[0]

    u_sampler = ConfidenceUncertaintySampler(xs_train, known_labels.copy(),
            sklearn.ensemble.RandomForestClassifier, {'random_state': 0})
    r_sampler = RandomSampler(xs_train, known_labels.copy(),
            sklearn.ensemble.RandomForestClassifier, {'random_state': 0})
    q_sampler = QBCSampler(xs_train, known_labels.copy(),
            sklearn.ensemble.RandomForestClassifier)

    def test_sampler(sampler):
        scores = []
        for i in range(400):
            index = sampler.sample_index()
            label = groundtruth_train[index]
            sampler.add_label(index, label)
            scores.append(sampler.score(xs_test, ts_test))
        return scores

    u_scores, r_scores, q_scores = map(
            test_sampler, [u_sampler, r_sampler, q_sampler])

    return u_scores, r_scores, q_scores


if __name__ == '__main__':
    with multiprocessing.pool.Pool(processes=8) as pool:
        all_scores = [pool.apply_async(run_tests, ()) for i in range(1)]
        all_scores = numpy.array([r.get() for r in all_scores])

    mean_scores = numpy.mean(all_scores, axis=0)
    u_scores, r_scores, q_scores = mean_scores

# u_scores = numpy.mean(all_u_scores, axis=0)
# r_scores = numpy.mean(all_r_scores, axis=0)
# q_scores = numpy.mean(all_q_scores, axis=0)

    plt.plot(u_scores)
    plt.plot(r_scores)
    plt.plot(q_scores)
    plt.legend(['uncertainty', 'random', 'qbc'])
    plt.show()
