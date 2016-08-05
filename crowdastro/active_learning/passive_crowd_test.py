import unittest

import numpy

from . import passive_crowd

# Most of the tests here are regression or shape tests. Shape in particular -
# vectorising is hard and error-prone.

class TestLogisticRegression(unittest.TestCase):

    def test_logistic_regression_vector(self):
        """logistic_regression works on vectors of data."""
        a = numpy.ones(10)
        b = 1
        x = numpy.ones(10)
        self.assertTrue(numpy.isclose(
                passive_crowd.logistic_regression(a, b, x),
                0.999983))

    def test_logistic_regression_matrix(self):
        """logistic_regression works on matrices of data."""
        a = numpy.ones(10)
        b = 1
        x = numpy.ones((20, 10)) * numpy.linspace(0, 1, 10)
        self.assertTrue(numpy.allclose(
                passive_crowd.logistic_regression(a, b, x),
                numpy.ones(20) * 0.997527))

class TestAnnotatorModel(unittest.TestCase):

    def test_annotator_model_vector(self):
        """annotator_model works on vectors of data."""
        w = numpy.linspace(0, 1, 10)
        g = -1
        x = numpy.linspace(0, 1, 10)
        y = 1
        z = 1
        self.assertTrue(numpy.isclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                0.92552988))

        z = 0
        self.assertTrue(numpy.isclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                0.0744701169))

    def test_annotator_model_matrix(self):
        """annotator_model works on matrices of data."""
        w = numpy.linspace(0, 1, 10)
        g = -1
        x = numpy.ones((20, 10)) * numpy.linspace(0, 1, 10)
        y = 1
        z = 1
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones(20) * 0.92552988))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (20,))

        z = 0
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones(20) * 0.0744701169))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (20,))

    def test_annotator_model_anno_vector(self):
        """annotator_model works on vectors of annotators and data vectors."""
        w = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 5, axis=0)
        g = -numpy.ones(5)
        x = numpy.linspace(0, 1, 10)
        y = numpy.ones(5)
        z = 1
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones(5) * 0.92552988))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (5,))

        z = 0
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones(5) * 0.0744701169))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (5,))

    def test_annotator_model_anno_vector(self):
        """annotator_model works on vectors of annotators and data matrices."""
        w = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 5, axis=0)
        g = -numpy.ones(5)
        x = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 20, axis=0)
        y = numpy.ones((5, 20))
        z = 1
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones((5, 20)) * 0.92552988))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (5, 20))

        z = 0
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones((5, 20)) * 0.0744701169))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (5, 20))
