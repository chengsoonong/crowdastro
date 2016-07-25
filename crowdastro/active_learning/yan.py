"""Yan et al. (2010) EM crowd passive learning algorithm.

Matthew Alger
The Australian National University
2016
"""

import itertools
import logging

import matplotlib.pyplot as plt
import numpy
import scipy.optimize
import sklearn.datasets

n_annotators, n_dim, n_samples = 4, 2, 100

x, z = sklearn.datasets.make_classification(
        n_samples=n_samples,
        n_features=n_dim,
        n_informative=n_dim,
        n_redundant=0,
        random_state=0)
# x = numpy.linspace(0, 1, n_samples).reshape((n_samples, n_dim))
# numpy.random.shuffle(x)
# z = x < 0.5
z = z.reshape((-1, 1))
y = z.repeat(n_annotators, axis=1).T
# Flip labels at random.
for t in range(n_annotators):
    to_flip = numpy.arange(n_samples)
    numpy.random.shuffle(to_flip)
    to_flip = to_flip[:n_samples // 10]  # 10%
    to_flip.sort()
    y[t, to_flip] = 1 - y[t, to_flip]
assert y.shape == (n_annotators, n_samples)


def logistic_regression(a, b, x):
    # assert a.shape == (n_dim,)
    # assert isinstance(b, float)
    # assert x.shape == (n_dim,)
    res = scipy.special.expit(x.dot(a) + b)
    return res


def annotator_model(w, g, x, y, z):
    # assert w.shape == (n_dim,)
    # assert isinstance(g, float)
    # assert x.shape == (n_dim,)
    eta = logistic_regression(w, g, x)
    # assert isinstance(eta, float)
    label_difference = numpy.abs(y - z)
    return (numpy.power(1 - eta, label_difference)
            * numpy.power(eta, 1 - label_difference))


def unpack(params):
    a = params[:n_dim]
    b = params[n_dim]
    w = params[n_dim+1:n_dim+1+n_annotators*n_dim].reshape(
            (n_annotators, n_dim))
    g = params[n_dim+1+n_annotators*n_dim:]
    return a, b, w, g


def pack(a, b, w, g):
    return numpy.hstack([a, [b], w.ravel(), g])


def em(x, y, epsilon=1e-8):
    """Expectation-maximisation algorithm from Yan et al. (2010)."""

    # Init.
    a = numpy.random.normal(size=(n_dim,))
    b = 0.0
    w = numpy.random.normal(size=(n_annotators, n_dim))
    g = numpy.random.normal(size=(n_annotators,))
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
            posterior *= logistic_regression(a, b, x[i])
            for t in range(n_annotators):
                posterior *= annotator_model(w[t], g[t], x[i], y[t, i], z)
            posteriors[i] = posterior

        # Repeat for p(z_i = 0 | x_i, y_i).
        posteriors_0 = numpy.zeros((n_samples,))
        for i in range(n_samples):
            z = 0
            # Use the old parameters to compute the posterior.
            posterior = 1
            posterior *= 1 - logistic_regression(a, b, x[i])
            for t in range(n_annotators):
                posterior *= annotator_model(w[t], g[t], x[i], y[t, i], z)
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
            a, b, w, g = unpack(params)
            expectation = 0
            for i in range(n_samples):
                for t in range(n_annotators):
                    p_z = posteriors[i]
                    p_z_0 = posteriors_0[i]
                    anno = annotator_model(w[t], g[t], x[i], y[t, i], 1)
                    anno_0 = annotator_model(w[t], g[t], x[i], y[t, i], 0)
                    post = logistic_regression(a, b, x[i])

                    expectation += numpy.log(post) * p_z
                    expectation += numpy.log((1 - post)) * p_z_0
                    expectation += numpy.log(anno) * p_z
                    expectation += numpy.log(anno_0) * p_z_0

            # Also need the gradients.
            dQ_da = numpy.zeros(a.shape + (n_samples,))
            dQ_db = numpy.zeros((n_samples,))
            dQ_dw = numpy.zeros(w.shape)
            dQ_dg = numpy.zeros(g.shape)
            for i in range(n_samples):
                dp = posteriors[i] - posteriors_0[i]
                dQ_db_i = dp * scipy.special.expit(-x[i].dot(a) - b) * \
                        (1 - scipy.special.expit(-x[i].dot(a) - b))
                dQ_da[:, i] = dQ_db_i * x[i]
                dQ_db[i] = dQ_db_i
                for t in range(n_annotators):
                    dQ_dg_t_i = (-1) ** y[t, i] * (-dp) * \
                        scipy.special.expit(-x[i].dot(w[t]) - g[t]) * \
                        (1 - scipy.special.expit(-x[i].dot(w[t]) - g[t]))
                    dQ_dw[t] += dQ_dg_t_i * x[i]
                    dQ_dg[t] += dQ_dg_t_i

            dQ_da = numpy.sum(dQ_da, axis=1)
            dQ_db = numpy.sum(dQ_db, axis=0)
            grad = pack(dQ_da, dQ_db, dQ_dw, dQ_dg)
            # logging.debug('Gradient: %s', grad)

            return -expectation, -grad

        theta = pack(a, b, w, g)
        theta_, fv, inf = scipy.optimize.fmin_l_bfgs_b(Q, x0=theta,
                                                       approx_grad=False)
        logging.info('Terminated with Q = %2f', fv)
        logging.info(inf['task'].decode('ascii'))
        # TODO(MatthewJA): Implement explicit gradients.
        a_, b_, w_, g_ = unpack(theta_)

        logging.info('Found new parameters - b: %f -> %f', b, b_)

        # Check convergence.
        dist = numpy.linalg.norm(a - a_) ** 2 + (b - b_) ** 2
        logging.debug('Distance: {:.02f}'.format(dist))
        if dist <= epsilon:
            return a_, b_, w_, g_

        a, b, w, g = a_, b_, w_, g_


if __name__ == '__main__':
    logging.root.setLevel(logging.DEBUG)
    logging.captureWarnings(True)
    a, b, _, _ = em(x, y)
    predictions = numpy.round(logistic_regression(a, b, x))
    # print(predictions[:15])
    # print(z.ravel()[:15])
    # print((predictions.ravel() == z.ravel()).mean())
    plt.subplot(2, 3, 1)
    plt.title('Groundtruth')
    plt.scatter(x[z.ravel() == 0, 0], x[z.ravel() == 0, 1], c='blue', marker='x')
    plt.scatter(x[z.ravel() == 1, 0], x[z.ravel() == 1, 1], c='red', marker='x')
    for t in range(n_annotators):
        if t >= 2:
            plt.subplot(2, 3, 2 + t + 1)
        else:
            plt.subplot(2, 3, 2 + t)
        plt.title('Annotator {}'.format(t + 1))
        plt.scatter(x[y[t] == 0, 0], x[y[t] == 0, 1], c='blue', marker='x')
        plt.scatter(x[y[t] == 1, 0], x[y[t] == 1, 1], c='red', marker='x')
    plt.subplot(2, 3, 4)
    plt.title('Predictions')
    plt.scatter(x[predictions.ravel() == 0, 0], x[predictions.ravel() == 0, 1], c='blue', marker='x')
    plt.scatter(x[predictions.ravel() == 1, 0], x[predictions.ravel() == 1, 1], c='red', marker='x')
    plt.show()
