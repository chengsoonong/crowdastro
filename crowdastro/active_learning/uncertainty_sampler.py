"""Active learning with uncertainty sampling.

Pool-based. Binary class labels.

Matthew Alger
The Australian National University
2016
"""

import numpy

from .sampler import Sampler


class ConfidenceUncertaintySampler(Sampler):
    """Pool-based active learning with uncertainty sampling, with uncertainty
    based on confidence."""

    def __init__(self, pool, labels, Classifier, classifier_params=None):
        """
        pool: (n_samples, n_features) array of partially labelled data points.
        labels: (n_samples,) masked array of binary labels.
        classifier: Binary classifier class implementing a sklearn interface.
        classifier_params: Parameters to pass to Classifier. Default None.
        """
        super().__init__(pool, labels, Classifier,
                         classifier_params=classifier_params)
        self.compute_uncertainties()

    def compute_uncertainties(self):
        """Finds uncertainties for all objects in the pool."""
        # To keep things simple, I'll use the (negative) proximity to the
        # decision boundary as the uncertainty. Note that the uncertainties
        # array is masked such that labelled points have no uncertainty.
        probs = self.classifier.predict_proba(self.pool)[:, 1]
        self.uncertainties = numpy.ma.masked_array(-numpy.abs(probs - 0.5),
                                                   mask=~self.labels.mask)

    def sample_index(self):
        """Finds index of the least certain unlabelled point."""
        index = self.uncertainties.argmax()
        return index

    def sample_indicies(self, n):
        """Finds indices of the n least certain unlabelled points."""
        indices = self.uncertainties.argsort()
        return indices[-n:]

    def retrain(self):
        """Retrains the classifier."""
        super().retrain()
        self.compute_uncertainties()
