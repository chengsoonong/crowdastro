"""Crowd learning functions.

Matthew Alger
The Australian National University
2016
"""

import collections

import numpy
import scipy.special


def majority_vote(y):
    """Computes the majority vote of a set of crowd labels.

    y: (n_annotators, n_examples) NumPy masked array of labels.
    -> (n_examples,) NumPy array of labels.
    """
    _, n_samples = y.shape
    mv = numpy.zeros((n_samples,))
    for i in range(n_samples):
        labels = y[:, i]

        if labels.mask is False:
            counter = collections.Counter(labels)
        else:
            counter = collections.Counter(labels[~labels.mask])

        if counter:
            mv[i] = max(counter, key=counter.get)
        else:
            # No labels for this data point.
            mv[i] = numpy.random.randint(2)  # ¯\_(ツ)_/¯
    return mv


def logistic_regression(w, x):
    """Logistic regression classifier model.

    w: Weights w. (n_features,) NumPy array
    x: Data point x_i. (n_features,) NumPy array
    -> float in [0, 1]
    """
    return scipy.special.expit(numpy.dot(x, w.T))
