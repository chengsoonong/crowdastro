"""Crowd label processing functions.

Matthew Alger
The Australian National University
2016
"""

import collections

import numpy
import scipy.special
import sklearn.metrics


def balanced_accuracy(y_true, y_pred):
    """Computes the balanced accuracy of a predictor.

    y_true: (n_examples,) array of true labels.
    y_pred: (n_examples,) (masked) array of predicted labels.
    -> float or None (if the balanced accuracy isn't defined).
    """
    if hasattr(y_pred, 'mask') and not isinstance(y_pred.mask, bool):
        cm = sklearn.metrics.confusion_matrix(
                y_true[~y_pred.mask], y_pred[~y_pred.mask]).astype(float)
    else:
        cm = sklearn.metrics.confusion_matrix(y_true, y_pred).astype(float)

    tp = cm[1, 1]
    n, p = cm.sum(axis=1)
    tn = cm[0, 0]
    if not n or not p:
        return None

    ba = (tp / p + tn / n) / 2
    return ba


def crowd_label(y, alphas, betas):
    """Simulates a crowd performing a labelling task.

    y: (n_examples,) array of true labels.
    alphas: (n_labellers,) array of labeller true positive rates.
    betas: (n_labellers,) array of labeller true negative rates.
    """
    n_labellers = len(alphas)
    assert n_labellers == len(betas)
    n_examples = len(y)
    labels = numpy.zeros((n_labellers, n_examples))
    for i, true_label in enumerate(y):
        for t in range(n_labellers):
            if true_label == 0:
                if numpy.random.random() <= betas[t]:
                    labels[t, i] = 0
                else:
                    labels[t, i] = 1
            else:
                if numpy.random.random() <= alphas[t]:
                    labels[t, i] = 1
                else:
                    labels[t, i] = 0
    mask = numpy.zeros(labels.shape)  # Fully observed labels.
    return numpy.ma.MaskedArray(labels, mask=mask)


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
