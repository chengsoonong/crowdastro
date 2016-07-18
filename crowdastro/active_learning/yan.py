"""Yan et al. (2010) EM crowd passive learning algorithm.

Matthew Alger
The Australian National University
2016
"""

import logging

import numpy
import scipy.optimize
import sklearn.datasets

n_annotators, n_dim, n_samples = 10, 2, 100

data, z = sklearn.datasets.make_classification(
    n_samples=n_samples,
    n_features=n_dim,
    n_informative=n_dim,
    n_redundant=0,
    random_state=0)
z = z.reshape((-1, 1))
y = z  # No noise.

def logistic_regression(a, b):
    return (1 + numpy.exp(-data.dot(a) - b)) ** (-1)

def annotator_model(w, g):
    e_t = logistic_regression(w, g)
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

def target(args, p_z):
    a_, b_, w_, g_ = unpack(args)

    # Optimisation target.
    assert n_dim + 1 + n_dim * n_annotators + n_annotators == args.shape[0]

    # [...] (n_samples x n_annotators)
    inner_expectation = (numpy.log(annotator_model(w_, g_))
                        + numpy.log(logistic_regression(a_, b_)))
    assert inner_expectation.shape == (n_samples, n_annotators)

    # E_~p(z_i)[...] (n_annotators)
    expectation = (inner_expectation * p_z).sum(axis=0)
    assert expectation.shape == (n_annotators,)

    return -expectation.sum(axis=0)

def em(epsilon=1e-5):
    a, b = numpy.zeros((n_dim, 1)), 0
    w, g = numpy.zeros((n_dim, n_annotators)), numpy.zeros((1, n_annotators))
    while True:  # Until convergence.
        # Expectation step.
        # ~p(z_i) propto product(p(y_i^(t) | x_i, z_i) p(z_i | x_i)
        #                        for t in range(n_annotators))
        # p(y_i^(t) | x_i, z_i) is the Bernoulli annotator model.
        # p(z_i | x_i) is logistic regression.

        # p(z_i | x_i) (n_samples x 1)
        p_z_x = logistic_regression(a, b)

        # p(y_i^(t) | x_i, z_i) (n_samples x n_annotators)
        p_y_x_z = annotator_model(w, g)

        # ~p(z_i)
        p_z = p_y_x_z.prod(axis=1).reshape((-1, 1)) * p_z_x

        # Maximisation step.
        # We want to minimise a, b, w, and g with respect to
        #   -E_~p(z_i)[log p(y_i^(t) | x_i, z_i) + log p(z_i | x_i)].
        # Yan et al. suggest LBFGS, which scipy implements.
        packed = numpy.hstack([a.ravel(), b, w.ravel(), g.ravel()])
        optimised = scipy.optimize.fmin_l_bfgs_b(target, packed,
                                                 approx_grad=True,
                                                 args=(p_z,))
        # TODO(MatthewJA): Implement the exact gradient.
        best, target_value, info = optimised
        a_, b_, w_, g_ = unpack(best)

        change = abs(a_ - a).sum() + abs(b_ - b)
        logging.debug('Change: %s', change)
        if change < epsilon:
            return a_, b_, w_, g_

        a, b, w, g = a_, b_, w_, g_

if __name__ == '__main__':
    logging.root.setLevel(logging.DEBUG)
    print(em())
