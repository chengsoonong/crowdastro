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


def logistic_regression(w, x):
    """Logistic regression classifier.

    w: Weights w. (n_features,) NumPy array
    x: Data point x_i. (n_features,) NumPy array
    -> float in [0, 1]
    """
    return scipy.special.expit(numpy.dot(x, w.T))


def majority_vote(y):
    """Computes majority vote of partially observed crowd labels."""
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


def max_alpha_step(m, y, y_mask):
    """Computes α based on μ.

    m: μ
    y: Array of crowd labels.
    y_mask: Mask of unobserved crowd labels.
    -> α
    """
    return numpy.dot(y, m) / (m.sum() + EPS)


def max_beta_step(m, y, y_mask):
    """Computes β based on μ.

    m: μ
    y: Array of crowd labels.
    y_mask: Mask of unobserved crowd labels.
    -> β
    """
    return numpy.dot(1 - y - y_mask, 1 - m) / ((1 - m).sum() + EPS)


def _hessian_inverse_multiply(x, H, g):
    return numpy.linalg.norm(H.dot(x) - g)


def max_w_step(m, x, mv, init_w=None, epsilon=1e-4, step_size=1e-4,
               max_iters=5000):
    """Computes w based on μ.

    m: μ
    x: (n_samples, n_features) NumPy array of examples.
    mv: Majority vote of labels.
    -> w
    """
    n_samples, n_features = x.shape

    # lr = sklearn.linear_model.LogisticRegression(class_weight='balanced',
    #                                              fit_intercept=False)
    # lr.fit(x, mv)
    # w = lr.coef_.ravel()
    if init_w is None:
        w = numpy.random.normal(size=(x.shape[1],))
    else:
        w = init_w

    for i in range(max_iters):
        lr = logistic_regression(w, x)
        g = numpy.dot(m - lr, x)

        H = numpy.zeros((n_features, n_features))
        for i in range(n_samples):
            H += -lr[i] * (1 - lr[i]) * numpy.outer(x[i], x[i])

        # Need to find H^{-1} g. Since there may be many features, this is
        # fastest if we minimise ||Hx - g|| for x.
        invHg = scipy.optimize.fmin_bfgs(_hessian_inverse_multiply,
                numpy.random.normal(size=w.shape), args=(H, g), disp=False)

        w_ = w - step_size * invHg

        dw = numpy.linalg.norm(w_ - w)

        logging.debug('Current value of delta w: %f', dw)

        if dw < epsilon:
            return w

        w = w_

    logging.warning('Optimisation of w failed to converge, delta w: %f', dw)
    return w


def exp_m_step(a, b, w, x, y, y_mask):
    """Computes expectation value of μ."""
    lr = logistic_regression(w, x)
    exp_a = numpy.ones((x.shape[0],))
    exp_b = numpy.ones((x.shape[0],))
    for t in range(a.shape[0]):
        for i in range(x.shape[0]):
            if y_mask[t, i]:
                continue

            exp_a *= a[t] ** y[t, i] * (1 - a[t]) ** (1 - y[t, i])
            exp_b *= b[t] ** (1 - y[t, i]) * (1 - b[t]) ** y[t, i]

    return exp_a * lr / (exp_a * lr + exp_b * (1 - lr) + EPS)


def likelihood(a, b, w, x, y, y_mask):
    """Computes the likelihood p(D | θ) = p({x, y} | α, β, w).

    a: α. (n_labellers,) NumPy array.
    b: β. (n_labellers,) NumPy array.
    w: Weights w. (n_features,) NumPy array
    x: (n_samples, n_features) NumPy array of examples.
    y: (n_labellers, n_samples) Array of crowd labels.
    y_mask: Mask of unobserved crowd labels.
    -> Likelihood float
    """
    exp_a = numpy.ones((x.shape[0],))
    exp_b = numpy.ones((x.shape[0],))
    for t in range(a.shape[0]):
        for i in range(x.shape[0]):
            if y_mask[t, i]:
                continue

            exp_a *= a[t] ** y[t, i] * (1 - a[t]) ** (1 - y[t, i])
            exp_b *= b[t] ** (1 - y[t, i]) * (1 - b[t]) ** y[t, i]
    exp_p = logistic_regression(w, x)

    return (exp_a * exp_p + exp_b * (1 - exp_p)).prod()


def train(x, y, epsilon=1e-5, restarts=5):
    """Trains the Raykar algorithm.

    x: (n_samples, n_features) NumPy array of examples.
    y: (n_labellers, n_samples) Masked NumPy array of crowd labels.
    epsilon: Convergence threshold. Default 1e-5.
    restarts: Number of random restarts. Default 5.
    -> α, β, w
    """
    results = []
    x_with_bias = numpy.hstack([x, numpy.ones((x.shape[0], 1))])
    for trial in range(restarts):
        logging.debug('Trial {}/{}'.format(trial + 1, restarts))
        a, b, w = _train(x, y, epsilon=epsilon)
        lh = likelihood(a, b, w, x_with_bias, y.filled(0), y.mask)
        results.append((lh, (a, b, w)))

    return max(results, key=lambda z: z[0])[1]


def _train(x, y, epsilon=1e-5):
    """Trains the Raykar algorithm.

    x: (n_samples, n_features) NumPy array of examples.
    y: (n_labellers, n_samples) Masked NumPy array of crowd labels.
    -> α, β, w
    """
    # Add a bias feature.
    x = numpy.hstack([x, numpy.ones((x.shape[0], 1))])

    n_samples, n_dim = x.shape
    n_labellers, n_samples_ = y.shape
    assert n_samples == n_samples_, 'Label array has wrong number of labels.'

    # Compute majority vote labels for initialisation.
    m = mv = majority_vote(y)

    # Convert y into a dense array and a mask. Then we can ignore the mask when
    # we don't need it and get nice fast code (numpy.ma is quite slow).
    y_mask = y.mask
    y = y.filled(0)

    step_size = 1e-5
    w = None

    while True:
        # Maximisation step.
        a = max_alpha_step(m, y, y_mask)
        b = max_beta_step(m, y, y_mask)
        w = max_w_step(m, x, mv, step_size=step_size, init_w=w)
        # step_size /= 1.1

        # Expectation step.
        m_ = exp_m_step(a, b, w, x, y, y_mask)

        logging.debug('Current value of delta mu: %f', numpy.linalg.norm(m_ - m))

        if numpy.linalg.norm(m_ - m) < epsilon:
            return a, b, w

        m = m_


def predict(w, x):
    x = numpy.hstack([x, numpy.ones((x.shape[0], 1))])
    return logistic_regression(w, x).round()
