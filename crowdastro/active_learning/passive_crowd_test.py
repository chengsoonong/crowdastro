import unittest

import numpy

from . import passive_crowd

# Most of the tests here are regression or shape tests. Shape in particular -
# vectorising is hard and error-prone.

class TestLogisticRegression(unittest.TestCase):

    def test_logistic_regression_vector(self):
        """logistic_regression works on vectors."""
        a = numpy.ones(10)
        b = 1
        x = numpy.ones(10)
        self.assertTrue(numpy.isclose(
            passive_crowd.logistic_regression(a, b, x),
            0.999983))

