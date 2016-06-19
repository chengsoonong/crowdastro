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
        """Finds index of the least certain unlabelled point."""
        unlabelled = self.labels.mask.nonzero()[0]

        if len(unlabelled):
            index = numpy.random.choice(unlabelled)
            return index

        return 0
