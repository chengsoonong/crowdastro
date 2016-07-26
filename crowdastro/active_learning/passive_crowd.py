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
import scipy.special
import sklearn.datasets
import sklearn.linear_model

n_annotators, n_dim, n_samples = 4, 2, 50
x, z = sklearn.datasets.make_classification(
        n_samples=n_samples,
        n_features=n_dim,
        n_informative=n_dim,
        n_redundant=0,
        random_state=100)
z = z.reshape((-1, 1))
y = z.repeat(n_annotators, axis=1).T
if True:
    # Annotator 1 is really bad at this.
    y[0, :] = numpy.ones((n_samples,))
    # Annotator 2 is wrong 10% of the time.
    indices = numpy.arange(n_samples)
    numpy.random.shuffle(indices)
    indices = indices[:n_samples // 10]
    indices.sort()
    y[1, indices] = 1 - y[1, indices]
    # Annotator 3 is good, but only if x[0] > 0.
    y[2, x[:, 0] <= 0] = numpy.round(
            numpy.random.uniform(size=((x[:, 0] <= 0).sum(),)))
    # Annotator 4 is wrong 30% of the time.
    indices = numpy.arange(n_samples)
    numpy.random.shuffle(indices)
    indices = indices[:n_samples * 3 // 10]
    indices.sort()
    y[3, indices] = 1 - y[3, indices]
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


def em(x, y, epsilon=1e-5):
    """Expectation-maximisation algorithm from Yan et al. (2010)."""

    # Init.
    # For our initial guess, we'll fit logistic regression to a majority vote.
    lr_ab = sklearn.linear_model.LogisticRegression()
    majority_y = numpy.zeros((n_samples,))
    for i in range(n_samples):
        labels = y[:, i]
        majority_y[i] = numpy.bincount(labels).argmax()
    lr_ab.fit(x, majority_y)
    a = lr_ab.coef_.ravel()
    b = lr_ab.intercept_[0]
    w = numpy.zeros((n_annotators, n_dim))
    g = numpy.zeros((n_annotators,))
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
                p_z = posteriors[i]
                p_z_0 = posteriors_0[i]
                for t in range(n_annotators):
                    anno = annotator_model(w[t], g[t], x[i], y[t, i], 1)
                    anno_0 = annotator_model(w[t], g[t], x[i], y[t, i], 0)
                    assert numpy.isclose(anno + anno_0, 1), anno + anno_0
                    post = logistic_regression(a, b, x[i])

                    assert numpy.isclose(p_z + p_z_0, 1), p_z + p_z_0

                    if numpy.isclose(post, 0) or numpy.isclose(anno, 0) or \
                            numpy.isclose(post, 1) or numpy.isclose(anno, 1):
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
                dQ_db_i = dp * scipy.special.expit(x[i].dot(a) + b) * \
                        (1 - scipy.special.expit(x[i].dot(a) + b))
                dQ_da[:, i] = dQ_db_i * x[i]
                dQ_db[i] = dQ_db_i
                for t in range(n_annotators):
                    dQ_dg_t_i = (-1) ** y[t, i] * (-dp) * \
                        scipy.special.expit(x[i].dot(w[t]) + g[t]) * \
                        (1 - scipy.special.expit(x[i].dot(w[t]) + g[t]))
                    dQ_dw[t] += dQ_dg_t_i * x[i]
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
        logging.info('Terminated with Q = %4f', fv)
        logging.info(inf['task'].decode('ascii'))
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
    a, b, w, g = em(x, y)
    print('w:', w)
    print('g:', g)
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
