"""Yan et al. (2010) EM crowd passive learning algorithm.

Matthew Alger
The Australian National University
2016
"""

import logging

import matplotlib.pyplot as plt
import numpy
import scipy.optimize
import scipy.special
import sklearn.datasets


def logistic_regression(data, a, b, z):
    lr = scipy.special.expit(data.dot(a) + b)
    if z == 1:
        return lr
    if z == 0:
        return 1 - lr
    raise ValueError('Unknown z label: {}'.format(z))


def annotator_model(data, w, g, z):
    e_t = logistic_regression(data, w, g, z)
    assert (e_t >= 0).all()
    assert (e_t <= 1).all()
    return (numpy.power(1 - e_t, numpy.abs(y - z))
           * numpy.power(e_t, 1 - numpy.abs(y - z)))


def unpack(args):
    # args is a 1D array [a b w g] with all elements flattened.
    # Reconstruct a, b, w, g.
    a_ = args[:n_dim].reshape((n_dim, 1))
    b_ = args[n_dim]
    w_ = args[n_dim + 1:n_dim + 1 + n_dim * n_annotators]
    w_ = w_.reshape((n_dim, n_annotators))
    g_ = args[n_dim + 1 + n_dim * n_annotators:]
    g_ = g_.reshape((1, n_annotators))
    return a_, b_, w_, g_


def target(args, data, p_z_0, p_z_1):
    a, b, w, g = unpack(args)

    # Optimisation target.
    assert n_dim + 1 + n_dim * n_annotators + n_annotators == args.shape[0]

    # E_~p(z_i)[...] (n_samples x n_annotators)
    # assert (annotator_model(data, w, g, 1) > 0).all()
    expectation = (numpy.log(annotator_model(data, w, g, 1)) * p_z_1
                  + numpy.log(annotator_model(data, w, g, 0)) * p_z_0
                  + numpy.log(logistic_regression(data, a, b, 1)) * p_z_1
                  + numpy.log(logistic_regression(data, a, b, 0)) * p_z_0)
    assert expectation.shape == (n_samples, n_annotators)

    return -expectation.sum()


def grad(args, data, p_z_0, p_z_1):
    a, b, w, g = unpack(args)

    dp = p_z_1 - p_z_0
    df_da = (dp * numpy.exp(-data.dot(a) - b) * data / (
             1 + numpy.exp(-data.dot(a) - b)) ** 2).sum(axis=0)
    df_db = (dp * numpy.exp(-data.dot(a) - b) / (
             1 + numpy.exp(-data.dot(a) - b)) ** 2).sum()
    df_dw = numpy.dot(data.T, logistic_regression(data, w, g, 1) * (
            1 - logistic_regression(data, w, g, 1)))
    df_dg = (logistic_regression(data, w, g, 1) * (
             1 - logistic_regression(data, w, g, 1))).sum(axis=0)

    grad_ = numpy.hstack([df_da.ravel(), df_db, df_dw.ravel(), df_dg.ravel()])
    assert grad_.shape == args.shape
    return -grad_


def em(data, y, z, epsilon=1e-6):
    n_samples, n_dim = data.shape
    n_annotators = y.shape[1]
    a, b = numpy.random.uniform(size=(n_dim, 1)), numpy.random.uniform()
    w = numpy.random.uniform(size=(n_dim, n_annotators))
    g = numpy.random.uniform(size=(1, n_annotators))

    while True:  # Until convergence.
        # Expectation step.
        # ~p(z_i) propto product(p(y_i^(t) | x_i, z_i) p(z_i | x_i)
        #                        for t in range(n_annotators))
        # p(y_i^(t) | x_i, z_i) is the Bernoulli annotator model.
        # p(z_i | x_i) is logistic regression.

        # p(z_i = 1 | x_i) (n_samples x 1)
        p_z_x_1 = logistic_regression(data, a, b, 1)

        # p(z_i = 0 | x_i) (n_samples x 1)
        p_z_x_0 = logistic_regression(data, a, b, 0)

        # p(y_i^(t) | x_i, z_i = 1) (n_samples x n_annotators)
        p_y_x_z_1 = annotator_model(data, w, g, 1)

        # p(y_i^(t) | x_i, z_i = 0) (n_samples x n_annotators)
        p_y_x_z_0 = annotator_model(data, w, g, 0)

        # ~p(z_i = 1)
        p_z_1 = p_y_x_z_1.prod(axis=1).reshape((-1, 1)) * p_z_x_1

        # ~p(z_i = 0)
        p_z_0 = p_y_x_z_0.prod(axis=1).reshape((-1, 1)) * p_z_x_0

        # Maximisation step.
        # We want to minimise a, b, w, and g with respect to
        #   -E_~p(z_i)[log p(y_i^(t) | x_i, z_i) + log p(z_i | x_i)].
        # Yan et al. suggest LBFGS, which scipy implements.
        packed = numpy.hstack([a.ravel(), b, w.ravel(), g.ravel()])
        optimised = scipy.optimize.fmin_l_bfgs_b(target, packed,
                                                 fprime=grad,
                                                 args=(data, p_z_1, p_z_0))
        # TODO(MatthewJA): Implement the exact gradient.
        best, target_value, info = optimised
        a_, b_, w_, g_ = unpack(best)

        change = abs(a_ - a).sum() + abs(b_ - b)
        logging.debug('Change: %s', change)
        if change < epsilon:
            return a_, b_, w_, g_

        a, b, w, g = a_, b_, w_, g_



def predict(data, a, b):
    return numpy.round(logistic_regression(data, a, b, 1))


if __name__ == '__main__':
    logging.root.setLevel(logging.DEBUG)

    n_annotators, n_dim, n_samples = 10, 2, 1000

    data, z = sklearn.datasets.make_classification(
        n_samples=n_samples,
        n_features=n_dim,
        n_informative=n_dim,
        n_redundant=0,
        random_state=0)
    z = z.reshape((-1, 1))
    y = numpy.repeat(z, n_annotators, axis=1)

    # Randomly flip labels.
    subset = numpy.random.uniform(size=y.shape) > 0.9
    y[subset] = 1 - y[subset]

    a, b, w, g = em(data, y, z)
    print((predict(data, a, b) == z).mean())
    # import sklearn.linear_model
    # lr = sklearn.linear_model.LogisticRegression()
    # lr.fit(data, z)
    # alr = lr.coef_.ravel()
    # plt.scatter(data[(z == 0).ravel(), 0], data[(z == 0).ravel(), 1], c='r')
    # plt.scatter(data[(z == 1).ravel(), 0], data[(z == 1).ravel(), 1], c='b')
    # xs = numpy.linspace(-4, 4, 10)
    # plt.plot(xs, a[0, 0] * xs + a[1, 0], c='g')
    # plt.plot(xs, alr[0] * xs + alr[1], c='orange')
    # plt.show()