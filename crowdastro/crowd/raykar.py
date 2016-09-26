"""Raykar et al. (2010) EM crowd learning algorithm.

Matthew Alger
The Australian National University
2016
"""

import logging
import time

import numpy
import scipy.optimize
import sklearn.linear_model

from crowdastro.crowd.util import majority_vote, logistic_regression

EPS = 1E-8


class RaykarClassifier(object):
    """Classifier based on the Raykar et al. (2010) EM algorithm.

    Jointly learns an annotator model and a classification model.
    """

    def __init__(self, n_restarts=5, epsilon=1e-5, lr_init=True):
        """
        n_restarts: Number of times to run the algorithm. Higher numbers improve
            chances of finding a global maximum likelihood solution.
        epsilon: Convergence threshold.
        lr_init: Whether to initialise w using logistic regression.
        """
        self.n_restarts = n_restarts
        self.epsilon = epsilon
        self.lr_init = lr_init

    def fit(self, X, Y):
        """
        X: (n_samples, n_features) NumPy array of data.
        Y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        if X.shape[0] != Y.shape[1]:
            raise ValueError('X and Y have different numbers of samples.')

        results = []
        for trial in range(self.n_restarts):
            logging.debug('Trial {}/{}'.format(trial + 1, self.n_restarts))
            a, b, w = self._fit_params(X, Y)
            self.a_, self.b_, self.w_ = a, b, w
            lh = self.score(X, Y)
            results.append((lh, (a, b, w)))

        a, b, w = max(results, key=lambda z: z[0])[1]
        self.a_ = a
        self.b_ = b
        self.w_ = w

    def _fit_params(self, x, y):
        """
        x: (n_samples, n_features) NumPy array of data (with no bias term).
        y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        # Add a bias feature.
        x = numpy.hstack([x, numpy.ones((x.shape[0], 1))])

        n_samples, n_dim = x.shape
        n_labellers, n_samples_ = y.shape
        if n_samples_ != n_samples:
            raise ValueError('x and y have different numbers of samples.')

        self.n_samples_ = n_samples
        self.n_labellers_ = n_labellers
        self.n_dim_ = n_dim

        # Compute majority vote labels for initialisation.
        mv = majority_vote(y)
        m = mv.copy()
        # Add a small random factor for variety.
        m[m == 1] -= numpy.abs(numpy.random.normal(scale=1e-2,
                                                   size=m[m == 1].shape[0]))
        m[m == 0] += numpy.abs(numpy.random.normal(scale=1e-2,
                                                   size=m[m == 0].shape[0]))

        # Convert y into a dense array and a mask. Then we can ignore the mask
        # when we don't need it and get nice fast code (numpy.ma is quite slow).
        y_mask = y.mask
        y_1 = y.filled(1)
        y = y_0 = y.filled(0)

        w = None
        a = self._max_alpha_step(m, y, y_mask)
        b = self._max_beta_step(m, y, y_mask)

        while True:
            then = time.time()
            # Maximisation step.
            # w first, so the optimisation uses the old parameters.
            w = self._max_w_step(a, b, m, x, y_0, y_1, mv, init_w=w)
            a = self._max_alpha_step(m, y, y_mask)
            b = self._max_beta_step(m, y, y_mask)

            # Expectation step.
            m_ = self._exp_m_step(a, b, w, x, y, y_mask)

            logging.debug('Current value of delta mu: %f',
                          numpy.linalg.norm(m_ - m))

            dm = numpy.linalg.norm(m_ - m)
            if dm < self.epsilon:
                logging.debug('a: {}'.format(a))
                logging.debug('b: {}'.format(b))
                return a, b, w

            m = m_

            # Estimate time remaining.
            now = time.time()
            dt = now - then
            logging.debug('Raykar iteration took {} s.'.format(dt))

    def _exp_m_step(self, a, b, w, x, y, y_mask):
        """Computes expectation value of μ."""
        lr = logistic_regression(w, x)
        exp_a = numpy.ones((x.shape[0],))
        exp_b = numpy.ones((x.shape[0],))
        for t in range(a.shape[0]):
            for i in range(x.shape[0]):
                if y_mask[t, i]:
                    continue

                exp_a[i] *= a[t] ** y[t, i] * (1 - a[t]) ** (1 - y[t, i])
                exp_b[i] *= b[t] ** (1 - y[t, i]) * (1 - b[t]) ** y[t, i]

        logging.debug('Average a_i: {:.02}'.format(exp_a.mean()))
        logging.debug('Average alpha_t: {:.02}'.format(a.mean()))
        logging.debug('Max alpha_t: {}'.format(a.max()))
        logging.debug('Min alpha_t: {}'.format(a.min()))

        return exp_a * lr / (exp_a * lr + exp_b * (1 - lr) + EPS)

    def _hessian_inverse_multiply(self, x, H, g):
        return numpy.linalg.norm(H.dot(x) - g)

    def _max_w_step(self, a, b, m, x, y_0, y_1, mv, init_w=None):
        """Computes w based on μ.

        m: μ
        x: (n_samples, n_features) NumPy array of examples.
        y_0, y_1: Array of crowd labels filled with 0 or 1.
        mv: Majority vote of labels.
        init_w: Initial value of w.
        -> w
        """
        n_samples, n_features = x.shape

        if init_w is None and not self.lr_init:
            w = numpy.random.normal(size=(x.shape[1],))
        elif init_w is None:
            lr = sklearn.linear_model.LogisticRegression(
                    class_weight='balanced', fit_intercept=False)
            lr.fit(x, mv)
            w = lr.coef_.ravel()
        else:
            w = init_w

        w = scipy.optimize.fmin_bfgs(self._log_likelihood, w,
                                     args=(a.reshape((-1, 1)),
                                           b.reshape((-1, 1)), x, y_0, y_1),
                                     disp=False)
        return w

    def _max_alpha_step(self, m, y, y_mask):
        """Computes α based on μ.

        m: μ
        y: Array of crowd labels.
        y_mask: Mask of unobserved crowd labels.
        -> α
        """
        a = numpy.zeros((y.shape[0],))
        divisor = numpy.zeros((y.shape[0],))
        for t in range(y.shape[0]):
            for i in range(y.shape[1]):
                if y_mask[t, i]:
                    continue

                a[t] += m[i] * y[t, i]
                divisor[t] += m[i]

        divisor[divisor == 0] = EPS
        return a / divisor

    def _max_beta_step(self, m, y, y_mask):
        """Computes β based on μ.

        m: μ
        y: Array of crowd labels.
        y_mask: Mask of unobserved crowd labels.
        -> β
        """
        b = numpy.zeros((y.shape[0],))
        divisor = numpy.zeros((y.shape[0],))
        for t in range(y.shape[0]):
            for i in range(y.shape[1]):
                if y_mask[t, i]:
                    continue

                b[t] += (1 - m[i]) * (1 - y[t, i])
                divisor[t] += (1 - m[i])

        divisor[divisor == 0] = EPS
        return b / divisor

    def predict(self, X):
        return self.predict_proba(X).round()

    def predict_proba(self, X):
        X = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
        return logistic_regression(self.w_, X)

    def score(self, X, Y):
        """Computes the likelihood of labels and data under the model.

        X: (n_samples, n_features) NumPy array of data.
        Y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        X = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
        return self._likelihood(self.w_, self.a_.reshape((-1, 1)),
                                self.b_.reshape((-1, 1)), X, Y.filled(0),
                                Y.filled(1))

    def _log_likelihood(self, *args, **kwargs):
        return numpy.log(self._likelihood(*args, **kwargs) + EPS)

    def _likelihood(self, w, a, b, X, Y_0, Y_1):
        """Computes the likelihood of labels and data under a model.

        X: (n_samples, n_features) NumPy array of data.
        Y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        n_examples = X.shape[0]
        exp_p = logistic_regression(w, X)
        exp_a = numpy.ones((n_examples,))
        exp_b = numpy.ones((n_examples,))
        exp_a = numpy.power(a, Y_0).prod(axis=0)
        exp_a *= numpy.power(1 - a, 1 - Y_1).prod(axis=0)
        exp_b *= numpy.power(b, 1 - Y_1).prod(axis=0)
        exp_b *= numpy.power(1 - b, Y_0).prod(axis=0)

        return (exp_a * exp_p.T + exp_b * (1 - exp_p).T).prod()

    def get_params(self, deep=True):
        return {
            'n_restarts': self.n_restarts,
            'epsilon': self.epsilon,
            'lr_init': self.lr_init,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def serialise(self):
        """Returns a NumPy array representing the optimised parameters."""
        return numpy.concatenate([
                [self.n_labellers_],
                self.a_.ravel(),
                self.b_.ravel(),
                self.w_.ravel(),
        ])

    @classmethod
    def unserialise(cls, array):
        """Converts a NumPy array into a RaykarClassifier."""
        rc = cls()
        n_annotators = int(array[0])
        array = array[1:]
        rc.a_ = array[:n_annotators]
        rc.b_ = array[n_annotators:n_annotators * 2]
        rc.w_ = array[n_annotators * 2:]
        return rc
