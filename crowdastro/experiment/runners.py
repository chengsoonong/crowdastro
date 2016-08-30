"""Functions to run experiments.

Matthew Alger
The Australian National University
2016
"""

import numpy
import sklearn.linear_model


def to_params(lr):
    """Converts a LogisticRegression object into an array."""
    return numpy.concatenate([lr.coef_.ravel(), lr.intercept_])


def lr(results, method_name, split_id, features, targets, test_indices, C=1.0):
    """Run logistic regression and store results.

    results: Results object.
    features: (n_examples, n_features) array of features.
    targets: (n_examples,) array of binary targets.
    test_indices: List of integer testing indices.
    C: Regularisation parameter. Default 1.0. Higher values mean less
        regularisation.
    method_name: Name of this method in the results.
    split_id: ID of the split in the results.
    """
    assert max(test_indices) < features.shape[0]
    assert min(test_indices) >= 0

    train_indices = sorted(set(range(features.shape[0])) - set(test_indices))

    lr = sklearn.linear_model.LogisticRegression(class_weight='balanced', C=C)
    lr.fit(features[train_indices], targets[train_indices])
    results.store_trial(method_name, split_id,
            lr.predict_proba(features[test_indices])[:, 1],
            indices=test_indices, params=to_params(lr))
