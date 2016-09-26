"""Unit tests for Raykar code.

Matthew Alger
The Australian National University
2016
"""

import unittest

import numpy
import scipy.optimize
import sklearn.linear_model

from crowdastro.crowd import raykar
from crowdastro.crowd.util import logistic_regression


class TestSymmetry(unittest.TestCase):
    """Tests that the classifier is symmetric in its labels.

    i.e. Swapping 0 and 1 doesn't change the output.
    """

    def setUp(self):
        self.n_labellers = 10
        self.n_features = 1
        self.n_examples = 50
        self.noise = 0.25
        self.masked = 0.50
        self.x = numpy.arange(self.n_examples).reshape((-1, 1))
        y = (self.x.ravel() < self.n_examples // 4).astype(int)
        self.z = y
        y = numpy.tile(y, (self.n_labellers, 1))
        to_flip = numpy.random.binomial(
            1, self.noise, size=y.shape).astype(bool)
        y[to_flip] = 1 - y[to_flip]
        y_mask = numpy.random.binomial(1, self.masked, size=y.shape)
        self.y = numpy.ma.MaskedArray(y, mask=y_mask)
        numpy.random.seed(0)
        self.rc = raykar.RaykarClassifier(
            n_restarts=5,
            epsilon=1e-5,
            lr_init=False)

    def test_likelihood(self):
        """_likelihood is symmetric."""
        lr = sklearn.linear_model.LogisticRegression(
            class_weight='balanced',
            fit_intercept=True)
        lr.fit(self.x, self.z)
        w = numpy.concatenate([lr.coef_, lr.intercept_.reshape((1, 1))], axis=1)
        x = numpy.hstack([self.x, numpy.ones((self.n_examples, 1))])
        a = numpy.ones((self.n_labellers, 1)) * self.noise
        b = numpy.ones((self.n_labellers, 1)) * self.noise
        y_0 = self.y.filled(0)
        y_1 = self.y.filled(1)
        lh = self.rc._likelihood(w, a, b, x, y_0, y_1)
        y_flip = 1 - self.y
        y_0 = y_flip.filled(0)
        y_1 = y_flip.filled(1)
        lr.fit(self.x, 1 - self.z)
        w = numpy.concatenate([lr.coef_, lr.intercept_.reshape((1, 1))], axis=1)
        lh_flip = self.rc._likelihood(w, b, a, x, y_0, y_1)
        self.assertTrue(numpy.isclose(lh, lh_flip))

    def test_max_steps(self):
        """_max_beta_step and _max_alpha_step are symmetric."""
        m = numpy.random.random(size=(self.n_examples,))
        mbs = self.rc._max_beta_step(m, self.y.filled(0), self.y.mask)
        mas = self.rc._max_alpha_step(m, self.y.filled(0), self.y.mask)
        mbs_ = self.rc._max_beta_step(1 - m, 1 - self.y.filled(1), self.y.mask)
        mas_ = self.rc._max_alpha_step(1 - m, 1 - self.y.filled(1), self.y.mask)
        self.assertTrue(numpy.allclose(mbs, mas_))
        self.assertTrue(numpy.allclose(mas, mbs_))

    def test_exp_m_step(self):
        """_exp_m_step is symmetric."""
        a = numpy.random.random(size=(self.n_labellers, 1))
        b = numpy.random.random(size=(self.n_labellers, 1))
        w = numpy.random.random(size=(self.n_features + 1))
        x = numpy.hstack([self.x, numpy.ones((self.n_examples, 1))])
        y_mask = self.y.mask
        m = self.rc._exp_m_step(a, b, w, x, self.y.filled(0), y_mask)
        w_flip = scipy.optimize.fmin_bfgs(
            lambda k: ((logistic_regression(w, x) - 1 +
                        logistic_regression(k, x)) ** 2 +
                       0.000001 * numpy.linalg.norm(k)).sum(), w, disp=False)
        m_ = self.rc._exp_m_step(b, a, w_flip, x, 1 - self.y.filled(1), y_mask)
        self.assertTrue(numpy.allclose(m, 1 - m_, atol=1e-3))


if __name__ == '__main__':
    unittest.main()
