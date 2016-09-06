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


def lr_to_params(lr):
    """Converts a LogisticRegression object into an array."""
    return numpy.concatenate([lr.coef_.ravel(), lr.intercept_])


def make_train_indices(test_indices, n_examples):
    """Makes a set of train indices from a set of test indices.

    test_indices: List of test indices.
    n_examples: Number of total (testing + training) examples.
    -> sorted list of training indices.
    """
    return sorted(set(range(n_examples)) - set(test_indices))


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
    results.store_trial(method_name, split_id,
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
    results.store_trial(method_name, split_id,
            rf.predict_proba(features[test_indices])[:, 1],
            indices=test_indices, params=numpy.zeros((results.n_params,)))


def raykar(results, method_name, split_id, features, targets, test_indices,
        overwrite=False):
    """Run the Raykar algorithm and store results.

    results: Results object.
    features: (n_examples, n_features) array of features.
    targets: (n_labellers, n_examples) masked array of binary targets.
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

    rc = RaykarClassifier()
    rc.fit(features[train_indices], targets[:, train_indices])
    results.store_trial(method_name, split_id,
            rc.predict_proba(features[test_indices])[:, 1],
            indices=test_indices, params=rc.serialise())

