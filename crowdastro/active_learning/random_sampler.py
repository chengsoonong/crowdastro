"""Learning with random sampling.

Pool-based. Binary class labels.

Matthew Alger
The Australian National University
2016
"""

import numpy

from .sampler import Sampler


class RandomSampler(Sampler):
    """Pool-based learning with random sampling."""

    def sample_index(self):
        """Finds index of a random unlabelled point."""
        unlabelled = self.labels.mask.nonzero()[0]

        if len(unlabelled):
            index = numpy.random.choice(unlabelled)
            return index

        return 0

    def sample_indices(self, n):
        """Finds indices of n random unlabelled points."""
        indices = set()
        unlabelled = self.labels.mask.nonzero()[0]

        if len(unlabelled) < n:
            return unlabelled

        while len(indices) < n:
            index = numpy.random.choice(unlabelled)
            indices.add(index)

        return sorted(indices)
