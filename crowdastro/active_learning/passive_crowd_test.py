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


class TestQ(unittest.TestCase):

    def test_q(self):
        """Q returns correct value and gradients."""
        n_dim = 10
        n_samples = 20
        n_annotators = 5
        posteriors = numpy.linspace(0, 1, n_samples)
        posteriors_0 = 1 - posteriors
        x = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 20, axis=0)
        y = numpy.ones((5, 20))
        a = numpy.ones(10)
        b = 1
        w = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 5, axis=0)
        g = -numpy.ones(5)
        params = passive_crowd.pack(a, b, w, g)
        value, grad = passive_crowd.Q(params, n_dim, n_annotators, n_samples,
                posteriors, posteriors_0, x, y)

        self.assertTrue(numpy.isclose(
                value,
                4.339848789199))
        self.assertTrue(numpy.allclose(
                grad,
                numpy.array([
                        -0.00000000e+00,   2.16840434e-21,   4.33680869e-21,
                         6.50521303e-21,   8.67361738e-21,   1.30104261e-20,
                         1.30104261e-20,   8.67361738e-21,   1.73472348e-20,
                         3.03576608e-20,   3.03576608e-20,  -0.00000000e+00,
                         9.54097912e-20,   1.90819582e-19,   3.46944695e-20,
                         3.81639165e-19,   4.85722573e-19,   6.93889390e-20,
                         5.55111512e-19,   7.63278329e-19,   1.11022302e-18,
                        -0.00000000e+00,   9.54097912e-20,   1.90819582e-19,
                         3.46944695e-20,   3.81639165e-19,   4.85722573e-19,
                         6.93889390e-20,   5.55111512e-19,   7.63278329e-19,
                         1.11022302e-18,  -0.00000000e+00,   9.54097912e-20,
                         1.90819582e-19,   3.46944695e-20,   3.81639165e-19,
                         4.85722573e-19,   6.93889390e-20,   5.55111512e-19,
                         7.63278329e-19,   1.11022302e-18,  -0.00000000e+00,
                         9.54097912e-20,   1.90819582e-19,   3.46944695e-20,
                         3.81639165e-19,   4.85722573e-19,   6.93889390e-20,
                         5.55111512e-19,   7.63278329e-19,   1.11022302e-18,
                        -0.00000000e+00,   9.54097912e-20,   1.90819582e-19,
                         3.46944695e-20,   3.81639165e-19,   4.85722573e-19,
                         6.93889390e-20,   5.55111512e-19,   7.63278329e-19,
                         1.11022302e-18,   1.11022302e-18,   1.11022302e-18,
                         1.11022302e-18,   1.11022302e-18,   1.11022302e-18])))
