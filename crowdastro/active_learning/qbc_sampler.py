"""Active learning with query-by-committee.

Pool-based. Binary class labels.

Matthew Alger
The Australian National University
2016
"""

import collections

import numpy
import sklearn.metrics

from .sampler import Sampler
from ..crowd.util import balanced_accuracy, majority_vote


class QBCSampler(Sampler):
    """Pool-based active learning with query-by-committee, with uncertainty
    based on confidence."""

    def __init__(self, pool, labels, Classifier, n_classifiers=20,
                 classifier_params=None):
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
        self.n_classifiers = n_classifiers

        self.train()
        self.compute_disagreement()

    def train(self):
        """Train a new committee."""
        self.classifiers = [self.Classifier(random_state=i,
                                            **self.classifier_params)
                            for i in range(self.n_classifiers)]
        for c in self.classifiers:
            c.fit(self.pool[~self.labels.mask], self.labels[~self.labels.mask])

    def compute_disagreement(self):
        """Finds disagreement for all objects in the pool."""
        labels = numpy.array([c.predict(self.pool) for c in self.classifiers])
        # Each column is the classifications of one data point.
        n_agree = labels.sum(axis=0)
        self.disagreement = numpy.ma.masked_array(
                numpy.abs(n_agree - self.n_classifiers // 2),
                mask=~self.labels.mask)

    def sample_index(self):
        """Finds index of the most disagreed upon unlabelled point."""
        index = self.disagreement.argmin()
        return index

    def sample_indices(self, n):
        """Finds indices of the top n most disagreed upon unlabelled points."""
        indices = self.disagreement.argsort()
        return indices[:n]

    def retrain(self):
        """Retrains the classifier."""
        super().retrain()
        self.compute_disagreement()

    def score(self, test_xs, test_ts):
        """Finds cross-entropy error on test data."""
        return numpy.mean([
                sklearn.metrics.log_loss(
                        test_ts, c.predict(test_xs))
                for c in self.classifiers])

    def ba(self, test_xs, test_ts):
        """Finds balanced accuracy on test data."""
        labels = numpy.zeros((self.n_classifiers, len(test_xs)))
        for c in range(self.n_classifiers):
            preds = self.classifiers[c].predict(test_xs)
            labels[c, :] = preds
        labels = numpy.ma.MaskedArray(labels, mask=numpy.zeros(labels.shape))
        return balanced_accuracy(test_ts, majority_vote(labels))
