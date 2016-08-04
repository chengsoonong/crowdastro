"""Yan et al. (2010) EM crowd passive learning algorithm.

Matthew Alger
The Australian National University
2016
"""

import itertools
import logging

import numpy
import scipy.optimize
import scipy.special
import sklearn.linear_model


def logistic_regression(a, b, x):
    """Logistic regression classifier.

    a: Weights α. (n_dim,) NumPy array
    b: Bias β. float
    x: Data point x_i. (n_dim,) NumPy array
    -> float in [0, 1]
    """
    res = scipy.special.expit(numpy.dot(x, a) + b)
    return res


def annotator_model(w, g, x, y, z):
    """Yan et al. (2010) Bernoulli annotator model.

    w: Annotator weights w_t. (n_dim,) NumPy array
    g: Annotator bias γ_t. (n_dim,) NumPy array
    x: Data point x_i. (n_dim,) NumPy array
    y: Label y_i^(t). int
    z: "True" label z_i. int
    -> float in [0, 1]
    """
    eta = logistic_regression(w, g, x) + 0.0001
    label_difference = numpy.abs(y - z)
    return (numpy.power(1 - eta, label_difference)
            * numpy.power(eta, 1 - label_difference))


def unpack(params, n_dim, n_annotators):
    """Unpacks an array of parameters in to a, b, w, and g."""
    a = params[:n_dim]
    b = params[n_dim]
    w = params[n_dim+1:n_dim+1+n_annotators*n_dim].reshape(
            (n_annotators, n_dim))
    g = params[n_dim+1+n_annotators*n_dim:]
    return a, b, w, g


def pack(a, b, w, g):
    """Packs a, b, w, and g into an array of parameters."""
    return numpy.hstack([a, [b], w.ravel(), g])


def train(x, y, epsilon=1e-5, lr_init=False, skip_zeros=False):
    """Expectation-maximisation algorithm from Yan et al. (2010).

    x: Data. (n_samples, n_dim) NumPy array
    y: Labels. (n_annotators, n_samples) NumPy array
    epsilon: Convergence threshold. Default 1e-5. float
    lr_init: Initialised with logistic regression. Default False.
    skip_zeros: Whether to detect and skip zero probabilities. Default False.
    """
    n_samples, n_dim = x.shape
    n_annotators, n_samples_ = y.shape
    assert n_samples == n_samples_, 'Label array has wrong number of labels.'

    # Compute majority vote labels (for debugging + logistic regression init).
    majority_y = numpy.zeros((n_samples,))
    for i in range(n_samples):
        labels = y[:, i]
        majority_y[i] = numpy.bincount(labels).argmax()

    if lr_init:
        # For our initial guess, we'll fit logistic regression to the majority
        # vote.
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

    iters = 0
    while True:  # Until convergence (checked later).
        iters += 1
        logging.debug('Iteration %d', iters)

        # Expectation step.
        # Posterior for each i. p(z_i = 1 | x_i, y_i).
        posteriors = numpy.zeros((n_samples,))
        for i in range(n_samples):
            z = 1
            # Use the old parameters to compute the posterior.
            posterior = 1
            posterior *= logistic_regression(a, b, x[i, :])
            for t in range(n_annotators):
                posterior *= annotator_model(w[t], g[t], x[i, :], y[t, i], z)
            posteriors[i] = posterior

        # Repeat for p(z_i = 0 | x_i, y_i).
        posteriors_0 = numpy.zeros((n_samples,))
        for i in range(n_samples):
            z = 0
            # Use the old parameters to compute the posterior.
            posterior = 1
            posterior *= 1 - logistic_regression(a, b, x[i, :])
            for t in range(n_annotators):
                posterior *= annotator_model(w[t], g[t], x[i, :], y[t, i], z)
            posteriors_0[i] = posterior

        # We want to normalise. We want p(z = 1) + p(z = 0) == 1.
        # Currently, p(z = 1) + p(z = 0) == q.
        # :. Divide p(z = 1) and p(z = 0) by q.
        total = posteriors + posteriors_0
        posteriors /= total
        posteriors_0 /= total
        assert numpy.allclose(posteriors, 1 - posteriors_0), \
                (posteriors, posteriors_0)

        # Maximisation step.
        def Q(params):
            a, b, w, g = unpack(params, n_dim, n_annotators)
            expectation = 0
            for i in range(n_samples):
                p_z = posteriors[i]
                p_z_0 = posteriors_0[i]
                for t in range(n_annotators):
                    anno = annotator_model(w[t], g[t], x[i, :], y[t, i], 1)
                    anno_0 = annotator_model(w[t], g[t], x[i, :], y[t, i], 0)
                    assert numpy.isclose(anno + anno_0, 1), anno + anno_0
                    post = logistic_regression(a, b, x[i, :])

                    assert numpy.isclose(p_z + p_z_0, 1), p_z + p_z_0

                    if (numpy.isclose(post, 0) or numpy.isclose(anno, 0) or 
                            numpy.isclose(post, 1) or numpy.isclose(anno, 1)):
                        logging.debug('Found zero probabilities.')
                        logging.debug('a: %s, b: %f', a, b)
                        logging.debug('Mean ~p(z): %f', posteriors.mean())
                        predictions = predict(a, b, x)
                        class_weight = predictions.mean()
                        logging.debug('Predicted mean class: %f', class_weight)
                        logging.debug('True mean class: %f', majority_y.mean())
                        if skip_zeros:
                            return 10000000, numpy.zeros(params.shape)

                    expectation += numpy.log(post) * p_z
                    expectation += numpy.log((1 - post)) * p_z_0
                    expectation += numpy.log(anno) * p_z
                    expectation += numpy.log(anno_0) * p_z_0

            expectation /= n_annotators * n_samples

            # Also need the gradients.
            dQ_da = numpy.zeros(a.shape + (n_samples,))
            dQ_db = numpy.zeros((n_samples,))
            dQ_dw = numpy.zeros(w.shape)
            dQ_dg = numpy.zeros(g.shape)
            for i in range(n_samples):
                dp = posteriors[i] - posteriors_0[i]
                # dQ_db_i = n_annotators * (posteriors[i] -
                # logistic_regression(a, b, x[i, :]))
                dQ_db_i = dp * scipy.special.expit(x[i, :].dot(a) + b) * \
                        (1 - scipy.special.expit(x[i, :].dot(a) + b))
                dQ_da[:, i] = dQ_db_i * x[i, :]
                dQ_db[i] = dQ_db_i
                for t in range(n_annotators):
                    dQ_dg_t_i = (-1) ** y[t, i] * (-dp) * \
                        scipy.special.expit(x[i, :].dot(w[t]) + g[t]) * \
                        (1 - scipy.special.expit(x[i, :].dot(w[t]) + g[t]))
                    # dQ_dg_t_i = (2 * posteriors[i] * y[t, i] -
                    # logistic_regression(w[t], g[t], x[i, :]) - y[t, i] +
                    # posteriors_0[i]) / 100
                    dQ_dw[t] += dQ_dg_t_i * x[i, :]
                    dQ_dg[t] += dQ_dg_t_i

            dQ_da = numpy.sum(dQ_da, axis=1)
            dQ_db = numpy.sum(dQ_db, axis=0)
            grad = pack(dQ_da, dQ_db, dQ_dw, dQ_dg)
            # logging.debug('Gradient: %s', grad)

            grad /= n_annotators * n_samples

            return -expectation, -grad

        theta = pack(a, b, w, g)
        theta_, fv, inf = scipy.optimize.fmin_l_bfgs_b(Q, x0=theta,
                                                       approx_grad=False)
        logging.debug('Terminated with Q = %4f', fv)
        logging.debug(inf['task'].decode('ascii'))
        a_, b_, w_, g_ = unpack(theta_, n_dim, n_annotators)

        logging.debug('Found new parameters - b: %f -> %f', b, b_)

        # Check convergence.
        dist = numpy.linalg.norm(a - a_) ** 2 + (b - b_) ** 2
        logging.debug('Distance: {:.02f}'.format(dist))
        if dist <= epsilon:
            return a_, b_, w_, g_

        a, b, w, g = a_, b_, w_, g_


def predict(a, b, x):
    """Classify data points using logistic regression.

    a: Weights α. (n_dim,) NumPy array
    b: Bias β. float
    x: Data points. (n_samples, n_dim) NumPy array.
    """
    return numpy.round(logistic_regression(a, b, x))
