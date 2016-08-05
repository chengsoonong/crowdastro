"""Yan et al. (2011) EM crowd active learning algorithm with adjustments from
Izbicki & Stern (2013).

Matthew Alger
The Australian National University
2016
"""

import collections
import itertools
import logging

import numpy
import scipy.optimize
import scipy.special
import sklearn.linear_model

from .passive_crowd import annotator_model
from .passive_crowd import logistic_regression
from .passive_crowd import pack
from .passive_crowd import predict
from .passive_crowd import unpack


def Q(params, n_dim, n_annotators, n_samples, posteriors, posteriors_0, x, y):
    """Maximisation step minimisation target."""
    a, b, w, g = unpack(params, n_dim, n_annotators)
    expectation = 0

    assert numpy.allclose(posteriors + posteriors_0, 1)
    annos = annotator_model(w, g, x, y, 1)  # masked
    annos_0 = annotator_model(w, g, x, y, 0)  # masked
    posts = logistic_regression(a, b, x)
    expectation += n_annotators * numpy.dot(
            numpy.log(posts), posteriors_0)
    expectation += n_annotators * numpy.dot(
            numpy.log((1 - posts)), posteriors_0)
    expectation += numpy.ma.sum(numpy.ma.dot(numpy.log(annos), posteriors))
    expectation += numpy.ma.sum(numpy.ma.dot(numpy.log(annos_0), posteriors_0))

    # Also need the gradients.
    dp = posteriors - posteriors_0
    logit_i = scipy.special.expit(x.dot(a) + b)
    dQ_db_i = dp * logit_i * (1 - logit_i)
    dQ_da = numpy.dot(dQ_db_i, x)
    dQ_db = dQ_db_i.sum()

    logit_t = scipy.special.expit(numpy.dot(w, x.T) + g.reshape((-1, 1)))
    dQ_dg_t_i = numpy.power(-1, y) * (-dp) * logit_t * (1 - logit_t)
    dQ_dw = dQ_dg_t_i.dot(x)
    dQ_dg = dQ_dg_t_i.sum(axis=1)

    grad = pack(dQ_da, dQ_db, dQ_dw, dQ_dg)

    grad /= n_annotators * n_samples

    return -expectation, -grad


def em_step(n_samples, n_annotators, n_dim, a, b, w, g, x, y):
    # Expectation step.
    # Posterior for each i. p(z_i = 1 | x_i, y_i).
    lr = logistic_regression(a, b, x)
    posteriors = lr.copy()
    posteriors *= numpy.ma.prod(annotator_model(w, g, x, y, 1), axis=0)

    # Repeat for p(z_i = 0 | x_i, y_i).
    posteriors_0 = 1 - lr
    posteriors_0 *= numpy.ma.prod(annotator_model(w, g, x, y, 0), axis=0)

    # We want to normalise. We want p(z = 1) + p(z = 0) == 1.
    # Currently, p(z = 1) + p(z = 0) == q.
    # :. Divide p(z = 1) and p(z = 0) by q.
    total = posteriors + posteriors_0
    posteriors /= total
    posteriors_0 /= total
    assert numpy.allclose(posteriors, 1 - posteriors_0), \
            (posteriors, posteriors_0)

    # Maximisation step.
    theta = pack(a, b, w, g)
    theta_, fv, inf = scipy.optimize.fmin_l_bfgs_b(Q, x0=theta,
            approx_grad=False, args=(n_dim, n_annotators, n_samples,
                                     posteriors, posteriors_0, x, y))
    logging.debug('Terminated with Q = %4f', fv)
    logging.debug(inf['task'].decode('ascii'))
    a_, b_, w_, g_ = unpack(theta_, n_dim, n_annotators)

    logging.debug('Found new parameters - b: %f -> %f', b, b_)

    return a_, b_, w_, g_


def train(x, y, epsilon=1e-5, lr_init=False, skip_zeros=False):
    """Expectation-maximisation algorithm from Yan et al. (2011).

    x: Data. (n_samples, n_dim) NumPy array
    y: Labels. (n_annotators, n_samples) NumPy masked array
    epsilon: Convergence threshold. Default 1e-5. float
    lr_init: Initialised with logistic regression. Default False.
    skip_zeros: Whether to detect and skip zero probabilities. Default False.
    """
    n_samples, n_dim = x.shape
    n_annotators, n_samples_ = y.shape
    assert n_samples == n_samples_, 'Label array has wrong number of labels.'

    logging.info('Initialising...')

    if lr_init:
        # For our initial guess, we'll fit logistic regression to the majority
        # vote.

        # Compute majority vote labels.
        majority_y = numpy.zeros((n_samples,))
        for i in range(n_samples):
            labels = y[:, i]

            if labels.mask is False:
                counter = collections.Counter(labels)
            else:
                counter = collections.Counter(labels[~labels.mask])

            if counter:
                majority_y[i] = max(counter, key=counter.get)
            else:
                # No labels for this data point.
                majority_y[i] = numpy.random.randint(2)  # ¯\_(ツ)_/¯

        lr_ab = sklearn.linear_model.LogisticRegression()
        lr_ab.fit(x, majority_y)
        a = lr_ab.coef_.ravel()
        b = lr_ab.intercept_[0]
    else:
        a = numpy.random.normal(size=(n_dim,))
        b = numpy.random.normal()
    w = numpy.zeros((n_annotators, n_dim))
    g = numpy.ones((n_annotators,))

    logging.debug('Initial a: %s', a)
    logging.debug('Initial b: %s', b)
    logging.debug('Initial w: %s', w)
    logging.debug('Initial g: %s', g)
    assert x.shape == (n_samples, n_dim)
    assert y.shape == (n_annotators, n_samples)

    logging.info('Iterating until convergence...')

    iters = 0
    while True:  # Until convergence (checked later).
        iters += 1
        logging.info('Iteration %d', iters)

        a_, b_, w_, g_ = em_step(
                n_samples, n_annotators, n_dim, a, b, w, g, x, y)

        # Check convergence.
        dist = numpy.linalg.norm(a - a_) ** 2 + (b - b_) ** 2
        logging.debug('Distance: {:.02f}'.format(dist))
        if dist <= epsilon:
            return a_, b_, w_, g_

        a, b, w, g = a_, b_, w_, g_
