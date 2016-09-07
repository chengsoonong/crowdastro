"""Raykar et al. (2010) EM crowd learning algorithm.

Matthew Alger
The Australian National University
2016
"""

import collections
import logging

import numpy
import scipy.optimize
import sklearn.linear_model

EPS = 1e-10


def majority_vote(y):
    _, n_samples = y.shape
    mv = numpy.zeros((n_samples,))
    for i in range(n_samples):
        labels = y[:, i]

        if labels.mask is False:
            counter = collections.Counter(labels)
        else:
            counter = collections.Counter(labels[~labels.mask])

        if counter:
            mv[i] = max(counter, key=counter.get)
        else:
            # No labels for this data point.
            mv[i] = numpy.random.randint(2)  # ¯\_(ツ)_/¯
    return mv


class RaykarClassifier(object):
    """Classifier based on the Raykar et al. (2010) EM algorithm.

    Jointly learns an annotator model and a classification model.
    """

    def __init__(self, n_restarts=5, epsilon=1e-5, inner_epsilon=1e-4,
                 inner_step=1e-4, max_inner_iters=5000, lr_init=True):
        """
        n_restarts: Number of times to run the algorithm. Higher numbers improve
            chances of finding a global maximum likelihood solution.
        epsilon: Convergence threshold.
        inner_epsilon: Convergence threshold for maximisation step.
        inner_step: Step size for maximisation step.
        max_inner_iters: Maximum number of iterations for maximisation step.
        lr_init: Whether to initialise w using logistic regression.
        """
        self.n_restarts = n_restarts
        self.epsilon = epsilon
        self.inner_epsilon = inner_epsilon
        self.inner_step = inner_step
        self.max_inner_iters = max_inner_iters
        self.lr_init = lr_init

    def fit(self, X, Y):
        """
        X: (n_samples, n_features) NumPy array of data.
        Y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        if X.shape[0] != Y.shape[1]:
            raise ValueError('X and Y have different numbers of samples.')

        results = []
        x_with_bias = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
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
        x: (n_samples, n_features) NumPy array of data.
        y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        # Add a bias feature.
        x = numpy.hstack([x, numpy.ones((x.shape[0], 1))])

        n_samples, n_dim = x.shape
        n_labellers, n_samples_ = y.shape
        if n_samples_ != n_samples:
            raise ValueError('x and y have different numbers of samples.')

        # Compute majority vote labels for initialisation.
        mv = self._majority_vote(y)
        m = mv.copy()
        # Add a small random factor for variety.
        m[m == 1] -= numpy.abs(numpy.random.normal(scale=1e-2,
                                                   size=m[m == 1].shape[0]))
        m[m == 0] += numpy.abs(numpy.random.normal(scale=1e-2,
                                                   size=m[m == 0].shape[0]))

        # Convert y into a dense array and a mask. Then we can ignore the mask
        # when we don't need it and get nice fast code (numpy.ma is quite slow).
        y_mask = y.mask
        y = y.filled(0)

        w = None

        while True:
            # Maximisation step.
            a = self._max_alpha_step(m, y, y_mask)
            b = self._max_beta_step(m, y, y_mask)
            w = self._max_w_step(a, b, m, x, y, y_mask, mv, init_w=w)

            # Expectation step.
            m_ = self._exp_m_step(a, b, w, x, y, y_mask)

            logging.debug('Current value of delta mu: %f',
                          numpy.linalg.norm(m_ - m))

            if numpy.linalg.norm(m_ - m) < self.epsilon:
                logging.debug('a: {}'.format(a))
                logging.debug('b: {}'.format(b))
                return a, b, w

            m = m_

    def _exp_m_step(self, a, b, w, x, y, y_mask):
        """Computes expectation value of μ."""
        lr = self._logistic_regression(w, x)
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

    def _max_w_step(self, a, b, m, x, y, y_mask, mv, init_w=None):
        """Computes w based on μ.

        m: μ
        x: (n_samples, n_features) NumPy array of examples.
        y: Array of crowd labels.
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
                                     args=(a, b, x, y, y_mask),
                                     disp=False)
        return w

        # for i in range(self.max_inner_iters):
        #     lr = self._logistic_regression(w, x)
        #     g = numpy.dot(m - lr, x)

        #     H = numpy.zeros((n_features, n_features))
        #     for i in range(n_samples):
        #         H += -lr[i] * (1 - lr[i]) * numpy.outer(x[i], x[i])

        #     # Need to find H^{-1} g. Since there may be many features, this is
        #     # fastest if we minimise ||Hx - g|| for x.
        #     invHg = scipy.optimize.fmin_bfgs(self._hessian_inverse_multiply,
        #             numpy.random.normal(size=w.shape), args=(H, g), disp=False)

        #     w_ = w - self.inner_step * invHg

        #     dw = numpy.linalg.norm(w_ - w)

        #     logging.debug('Current value of delta w: %f', dw)

        #     if dw < self.inner_epsilon:
        #         return w

        #     w = w_

        # logging.warning('Optimisation of w failed to converge, delta w: %f', dw)
        # return w

    def _max_alpha_step(self, m, y, y_mask):
        """Computes α based on μ.

        m: μ
        y: Array of crowd labels.
        y_mask: Mask of unobserved crowd labels.
        -> α
        """
        logging.debug('Percentage y == m == 1: {:.02%}'.format(
                numpy.logical_and(y == 1, y == m.round()).mean()))
        logging.debug('Percentage m == 1: {:.02%}'.format(m.round().mean()))
        return numpy.dot(y, m) / (m.sum() + EPS)

    def _majority_vote(self, y):
        """Computes majority vote of partially observed crowd labels."""
        return majority_vote(y)

    def _logistic_regression(self, w, x):
        """Logistic regression classifier model.

        w: Weights w. (n_features,) NumPy array
        x: Data point x_i. (n_features,) NumPy array
        -> float in [0, 1]
        """
        return scipy.special.expit(numpy.dot(x, w.T))

    def _max_beta_step(self, m, y, y_mask):
        """Computes β based on μ.

        m: μ
        y: Array of crowd labels.
        y_mask: Mask of unobserved crowd labels.
        -> β
        """
        return numpy.dot(1 - y - y_mask, 1 - m) / ((1 - m).sum() + EPS)

    def predict(self, X):
        return self.predict_proba(X).round()

    def predict_proba(self, X):
        X = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
        return self._logistic_regression(self.w_, X)

    def score(self, X, Y):
        """Computes the likelihood of labels and data under the model.

        X: (n_samples, n_features) NumPy array of data.
        Y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        X = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
        return self._likelihood(self.w_, self.a_, self.b_, X, Y.filled(0),
                                Y.mask)

    def _log_likelihood(self, *args, **kwargs):
        return numpy.log(self._likelihood(*args, **kwargs) + EPS)

    def _likelihood(self, w, a, b, X, Y, y_mask):
        """Computes the likelihood of labels and data under a model.

        X: (n_samples, n_features) NumPy array of data.
        Y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        exp_a = numpy.ones((X.shape[0],))
        exp_b = numpy.ones((X.shape[0],))
        for t in range(a.shape[0]):
            for i in range(X.shape[0]):
                if y_mask[t, i]:
                    continue

                exp_a[i] *= a[t] ** Y[t, i] * (1 - a[t]) ** (1 - Y[t, i])
                exp_b[i] *= b[t] ** (1 - Y[t, i]) * (1 - b[t]) ** Y[t, i]
        exp_p = self._logistic_regression(w, X)

        return (exp_a * exp_p + exp_b * (1 - exp_p)).prod()

    def get_params(self, deep=True):
        return {
            'n_restarts': self.n_restarts,
            'epsilon': self.epsilon,
            'inner_epsilon': self.inner_epsilon,
            'inner_step': self.inner_step,
            'max_inner_iters': self.max_inner_iters,
            'lr_init': self.lr_init,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def serialise(self):
        """Returns a NumPy array representing the optimised parameters."""
        return numpy.concatenate([
                self.a_.ravel(),
                self.b_.ravel(),
                self.w_.ravel(),
        ])
