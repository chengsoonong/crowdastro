"""Active learning sampler base class.

Pool-based. Binary class labels.

Matthew Alger
The Australian National University
2016
"""

import numpy
import sklearn.metrics


class Sampler(object):
    """Pool-based active learning base class."""

    def __init__(self, pool, labels, Classifier, classifier_params=None):
        """
        pool: (n_samples, n_features) array of partially labelled data points.
        labels: (n_samples,) masked array of binary labels.
        classifier: Binary classifier class implementing a sklearn interface.
        classifier_params: Parameters to pass to Classifier. Default None.
        """
        self.pool = pool
        self.labels = labels
        self.Classifier = Classifier
        self.classifier_params = classifier_params or {}

        self.train()

    def sample_index(self):
        """Finds index of the unlabelled point to sample."""
        raise NotImplementedError()

    def add_label(self, index, label, retrain=True):
        """Adds a label from an oracle.

        index: Index of data point to label.
        label: Label from the oracle.
        """
        self.labels[index] = label
        if retrain:
            self.retrain()

    def train(self):
        """Trains the classifier."""
        self.classifier = self.Classifier(**self.classifier_params)
        self.classifier.fit(self.pool[~self.labels.mask],
                            self.labels[~self.labels.mask])

    def retrain(self):
        """Retrains the classifier."""
        # TODO(MatthewJA): Not sure if we should use warm starts here, so for
        # now I won't.
        self.train()

    def score(self, test_xs, test_ts):
        """Finds cross-entropy error on test data."""
        return sklearn.metrics.log_loss(
            test_ts,
            self.classifier.predict(test_xs))
