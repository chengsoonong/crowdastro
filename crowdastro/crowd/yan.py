"""Yan et al. (2010) EM crowd learning algorithm.

Matthew Alger
The Australian National University
2016
"""

import logging

import numba
import numpy
import scipy.optimize
import scipy.special
import sklearn.linear_model

from crowdastro.crowd.util import majority_vote as compute_majority_vote
from crowdastro.crowd.util import logistic_regression


EPS = 1E-8


class YanClassifier(object):
    """Classifier based on the Yan et al. (2010) EM algorithm.

    Jointly learns an annotator model and a classification model.
    """

    def __init__(self, epsilon=1e-5, lr_init=True, n_restarts=5):
        """
        epsilon: Convergence threshold. Default 1e-5.
        lr_init: Whether to initialise with logistic regression. Default True.
        n_restarts: Number of random restarts. Default 5.
        """
        self.epsilon = epsilon
        self.lr_init = lr_init
        self.n_restarts = n_restarts

    def fit(self, X, Y):
        """Fits parameters.

        X: (n_samples, n_features) NumPy array of data.
        Y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        if X.shape[0] != Y.shape[1]:
            raise ValueError('X and Y have different numbers of samples.')

        results = []
        for trial in range(self.n_restarts):
            logging.debug('Trial {}/{}'.format(trial + 1, self.n_restarts))
            a, w = self._fit_params(X, Y)
            self.a_, self.w_ = a, w
            lh = self.score(X, Y)
            results.append((lh, (a, w)))

        a, w = max(results, key=lambda z: z[0])[1]
        self.a_ = a
        self.w_ = w

    def _fit_params(self, X, Y):
        """Fits parameters.

        X: (n_samples, n_features) NumPy array of data.
        Y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        X = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
        n_samples, n_dim = X.shape
        n_annotators, n_samples_ = Y.shape
        assert n_samples == n_samples_, \
            'Label array has wrong number of labels.'
        majority_y = compute_majority_vote(Y)

        logging.debug('Initialising Yan method...')
        if self.lr_init:
            # For our initial guess, we'll fit logistic regression to the
            # majority vote.
            lr_ab = sklearn.linear_model.LogisticRegression(fit_intercept=False)
            lr_ab.fit(X, majority_y)
            a = lr_ab.coef_.ravel()
        else:
            a = numpy.random.normal(size=(n_dim,))
        w = numpy.random.normal(size=(n_annotators, n_dim))

        logging.debug('Initial a: %s', a)
        logging.debug('Initial w: %s', w)
        logging.debug('Iterating Yan method until convergence...')

        iters = 0
        while True:  # Until convergence (checked later).
            iters += 1
            logging.info('Iteration %d', iters)

            a_, w_ = self._em_step(n_samples, n_annotators, n_dim, a, w, X, Y)

            # Check convergence.
            dist = numpy.linalg.norm(a - a_) ** 2
            logging.debug('Distance: {}'.format(dist))
            if dist <= self.epsilon:
                return a_, w_

            a, w = a_, w_

    def _annotator_model(self, w, x, y, z):
        """Yan et al. (2010) Bernoulli annotator model.

        w: Annotator weights w_t. (n_dim,) NumPy array
        x: Data point x_i. (n_dim,) NumPy array
        y: Label y_i^(t). int
        z: "True" label z_i. int
        -> float in [0, 1]
        """
        eta = logistic_regression(w, x)
        label_difference = numpy.abs(y - z)
        return (numpy.power(1 - eta, label_difference.T) *
                numpy.power(eta, 1 - label_difference.T)).T

    def _unpack(self, params, n_dim, n_annotators):
        """Unpacks an array of parameters into a and w."""
        a = params[:n_dim]
        w = params[n_dim:].reshape((n_annotators, n_dim))
        return a, w

    def _pack(self, a, w):
        """Packs a and w into an array of parameters."""
        return numpy.hstack([a, w.ravel()])

    def _Q(self, params, n_dim, n_annotators, n_samples, posteriors,
           posteriors_0, x, y):
        """Maximisation step minimisation target."""
        a, w = self._unpack(params, n_dim, n_annotators)

        expectation = (
                posteriors.dot(
                        (numpy.log(self._annotator_model(w, x, y, 1) + EPS) +
                         numpy.log(logistic_regression(a, x) + EPS)).T) +
                posteriors_0.dot(
                        (numpy.log(self._annotator_model(w, x, y, 0) + EPS) +
                         numpy.log(1 - logistic_regression(a, x) + EPS)).T)
        ).sum()

        # Also need the gradients.
        dQ_da = n_annotators * (
                numpy.dot(posteriors * logistic_regression(-a, x) +
                          posteriors_0 * (logistic_regression(-a, x) - 1),
                          x))

        dQ_dw = numpy.zeros(w.shape)
        # Inefficient, but unrolled for clarity.
        # TODO(MatthewJA): Speed this up. (Numba?)
        for t in range(n_annotators):
            dQ_dw[t] += sum(
                    x[i] * posteriors[i] *
                    (logistic_regression(-w[t], x[i]) - abs(y[t, i] - 1)) +
                    x[i] * posteriors_0[i] *
                    (logistic_regression(-w[t], x[i]) - abs(y[t, i] - 0))
                    for i in range(n_samples))
        grad = self._pack(dQ_da, dQ_dw)

        return -expectation, -grad

    def _em_step(self, n_samples, n_annotators, n_dim, a, w, x, y):
        # Expectation step.
        # Posterior for each i. p(z_i = 1 | x_i, y_i).
        lr = logistic_regression(a, x)
        posteriors = lr.copy()
        posteriors *= self._annotator_model(w, x, y, 1).prod(axis=0)

        # Repeat for p(z_i = 0 | x_i, y_i).
        posteriors_0 = 1 - lr
        posteriors_0 *= self._annotator_model(w, x, y, 0).prod(axis=0)

        # We want to normalise. We want p(z = 1) + p(z = 0) == 1.
        # Currently, p(z = 1) + p(z = 0) == q.
        # :. Divide p(z = 1) and p(z = 0) by q.
        total = posteriors + posteriors_0
        posteriors /= total
        posteriors_0 /= total
        assert numpy.allclose(posteriors, 1 - posteriors_0), \
                             (posteriors, posteriors_0)

        # Maximisation step.
        theta = self._pack(a, w)
        theta_, fv, inf = scipy.optimize.fmin_l_bfgs_b(
                self._Q, x0=theta, approx_grad=False,
                args=(n_dim, n_annotators, n_samples, posteriors, posteriors_0,
                      x, y))
        logging.debug('Terminated with Q = %4f', fv)
        logging.debug(inf['task'].decode('ascii'))
        a_, w_, = self._unpack(theta_, n_dim, n_annotators)

        return a_, w_

    def predict(self, x):
        """Classify data points using logistic regression.

        x: Data points. (n_samples, n_dim) NumPy array.
        """
        return numpy.round(self.predict_proba(x))

    def predict_proba(self, x):
        """Predict probabilities of data points using logistic regression.

        x: Data points. (n_samples, n_dim) NumPy array.
        """
        x = numpy.hstack([x, numpy.ones((x.shape[0], 1))])
        return logistic_regression(self.a_, x)

    def score(self, X, Y):
        """Computes the likelihood of labels and data under the model.

        X: (n_samples, n_features) NumPy array of data.
        Y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        X = numpy.hstack([X, numpy.ones((X.shape[0], 1))])
        return self._likelihood(self.a_, self.w_, X, Y)

    def _log_likelihood(self, *args, **kwargs):
        return numpy.log(self._likelihood(*args, **kwargs) + EPS)

    @numba.jit
    def _likelihood(self, a, w, X, Y):
        """Computes the likelihood of labels and data under a model.

        X: (n_samples, n_features) NumPy array of data.
        Y: (n_labellers, n_samples) NumPy masked array of crowd labels.
        """
        lh = 1
        for i in range(X.shape[0]):
            for t in range(Y.shape[0]):
                if Y.mask[t, i]:
                    continue

                lr = logistic_regression(a, X[i])
                p1 = self._annotator_model(w[t], X[i], Y[t, i], 1) * lr
                p0 = self._annotator_model(w[t], X[i], Y[t, i], 0) * (1 - lr)
                lh *= p1 + p0

        return lh
