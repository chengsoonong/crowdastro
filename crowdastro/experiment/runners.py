"""Functions to run experiments.

Matthew Alger
The Australian National University
2016
"""

import logging

import numpy
import sklearn.ensemble
import sklearn.linear_model

from ..crowd.raykar import RaykarClassifier
from ..crowd.util import majority_vote
from ..crowd.yan import YanClassifier


def lr_to_params(lr):
    """Converts a LogisticRegression object into an array."""
    return numpy.concatenate([lr.coef_.ravel(), lr.intercept_])


def make_train_indices(test_indices, n_examples, downsample=False, labels=None):
    """Makes a set of train indices from a set of test indices.

    test_indices: List of test indices.
    n_examples: Number of total (testing + training) examples.
    downsample: Whether to downsample. Default False.
    labels: Binary labels. Only needed if downsample is True.
    -> sorted list of training indices.
    """
    if downsample and labels is None:
        raise ValueError('labels must be specified if downsample is specified')

    training = sorted(set(range(n_examples)) - set(test_indices))

    if not downsample:
        logging.debug('Not downsampling...')
        return training

    logging.debug('Downsampling...')

    training = numpy.array(training)

    positives = training[labels[training] == 1]
    negatives = training[labels[training] == 0]

    if positives.shape[0] > negatives.shape[0]:
        numpy.random.shuffle(positives)
        positives = positives[:negatives.shape[0]]
        training = numpy.concatenate([positives, negatives])
    else:
        numpy.random.shuffle(negatives)
        negatives = negatives[:positives.shape[0]]
        training = numpy.concatenate([negatives, positives])
    training.sort()

    logging.debug('Positive examples: {}'.format(positives.shape[0]))
    logging.debug('Negative examples: {}'.format(negatives.shape[0]))
    logging.debug('Total examples: {}'.format(training.shape[0]))
    assert training.shape[0] == positives.shape[0] + negatives.shape[0]
    return training


def lr(results, method_name, split_id, features, targets, test_indices, C=1.0,
       overwrite=False):
    """Run logistic regression and store results.

    results: Results object.
    features: (n_examples, n_features) array of features.
    targets: (n_examples,) array of binary targets.
    test_indices: List of integer testing indices.
    method_name: Name of this method in the results.
    split_id: ID of the split in the results.
    C: Regularisation parameter. Default 1.0. Higher values mean less
        regularisation.
    overwrite: Whether to overwrite existing results (default False).
    """
    assert max(test_indices) < features.shape[0]
    assert min(test_indices) >= 0

    train_indices = make_train_indices(test_indices, features.shape[0])

    if results.has_run(method_name, split_id) and not overwrite:
        logging.info('Skipping trial {}:{}.'.format(method_name, split_id))
        return

    lr = sklearn.linear_model.LogisticRegression(class_weight='balanced', C=C)
    lr.fit(features[train_indices], targets[train_indices])
    results.store_trial(
        method_name, split_id,
        lr.predict_proba(features[test_indices])[:, 1],
        indices=test_indices, params=lr_to_params(lr))


def rf(results, method_name, split_id, features, targets, test_indices,
       overwrite=False):
    """Run random forest and store results.

    Does not store the model as random forests are difficult to serialise.

    results: Results object.
    features: (n_examples, n_features) array of features.
    targets: (n_examples,) array of binary targets.
    test_indices: List of integer testing indices.
    method_name: Name of this method in the results.
    split_id: ID of the split in the results.
    overwrite: Whether to overwrite existing results (default False).
    """
    assert max(test_indices) < features.shape[0]
    assert min(test_indices) >= 0

    train_indices = make_train_indices(test_indices, features.shape[0])

    if results.has_run(method_name, split_id) and not overwrite:
        logging.info('Skipping trial {}:{}.'.format(method_name, split_id))
        return

    rf = sklearn.ensemble.RandomForestClassifier(class_weight='balanced')
    rf.fit(features[train_indices], targets[train_indices])
    results.store_trial(
        method_name, split_id,
        rf.predict_proba(features[test_indices])[:, 1],
        indices=test_indices, params=numpy.zeros((results.n_params,)))


def raykar(results, method_name, split_id, features, targets, test_indices,
           overwrite=False, n_restarts=5, downsample=False):
    """Run the Raykar algorithm and store results.

    results: Results object.
    features: (n_examples, n_features) array of features.
    targets: (n_labellers, n_examples) masked array of binary targets.
    test_indices: List of integer testing indices.
    method_name: Name of this method in the results.
    split_id: ID of the split in the results.
    overwrite: Whether to overwrite existing results (default False).
    n_restarts: Number of random restarts. Default 5.
    downsample: Whether to downsample. Default False.
    """
    assert max(test_indices) < features.shape[0]
    assert min(test_indices) >= 0

    train_indices = make_train_indices(test_indices, features.shape[0],
                                       downsample=downsample,
                                       labels=majority_vote(targets))

    if results.has_run(method_name, split_id) and not overwrite:
        logging.info('Skipping trial {}:{}.'.format(method_name, split_id))
        return

    rc = RaykarClassifier(max_inner_iters=5, n_restarts=n_restarts)
    rc.fit(features[train_indices], targets[:, train_indices])
    results.store_trial(
        method_name, split_id,
        rc.predict_proba(features[test_indices]),
        indices=test_indices, params=rc.serialise())


def yan(results, method_name, split_id, features, targets, test_indices,
        overwrite=False, n_restarts=5):
    """Run the Yan algorithm and store results.

    results: Results object.
    features: (n_examples, n_features) array of features.
    targets: (n_labellers, n_examples) masked array of binary targets.
    test_indices: List of integer testing indices.
    method_name: Name of this method in the results.
    split_id: ID of the split in the results.
    overwrite: Whether to overwrite existing results (default False).
    n_restarts: Number of random restarts. Default 5.
    """
    assert max(test_indices) < features.shape[0]
    assert min(test_indices) >= 0

    train_indices = make_train_indices(test_indices, features.shape[0])

    if results.has_run(method_name, split_id) and not overwrite:
        logging.info('Skipping trial {}:{}.'.format(method_name, split_id))
        return

    yc = YanClassifier(n_restarts=n_restarts)
    yc.fit(features[train_indices], targets[:, train_indices])
    results.store_trial(method_name, split_id,
            yc.predict_proba(features[test_indices]),
            indices=test_indices, params=yc.serialise())

