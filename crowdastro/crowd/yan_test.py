import unittest

import numpy

from . import yan


class TestGradients(unittest.TestCase):
    """Tests analytic gradients match approximate gradients."""

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)

    def setUp(self):
        self.T = T = 4
        self.D = D = 5
        self.N = N = 20

        self.x = numpy.random.random(size=(N, D))
        self.x_with_bias = numpy.hstack(
                [self.x, numpy.ones((self.x.shape[0], 1))])
        self.y = numpy.random.binomial(1, 0.5, size=(T, N))

        self.posteriors = numpy.random.random(size=(N,))
        self.posteriors_0 = 1 - self.posteriors

        self.params = numpy.random.normal(scale=0.5, size=(D + 1 + T * D + T,))

        self.yc = yan.YanClassifier()
        self.grad = yan.YanClassifier._Q(
                self.yc,
                self.params, D + 1, T, N, self.posteriors, self.posteriors_0,
                self.x_with_bias, self.y)[1]

        self.trials = 10

    # A neater way to do this would be to just loop over all indices in the
    # gradient. I've chosen to split this into each parameter so that the tests
    # will better indicate which gradients are incorrect.

    def test_grad_w(self):
        for t in range(self.T):
            for d in range(self.D):
                grads = []
                index = self.D+1+self.D*t+d
                for i in range(self.trials):
                    h = numpy.zeros(self.params.shape)
                    h[index] = (i + 1) / self.trials * \
                        numpy.linalg.norm(self.params) * 1e-10
                    grads.append((yan.YanClassifier._Q(
                                self.yc,
                                self.params + h,
                                self.D + 1,
                                self.T,
                                self.N,
                                self.posteriors,
                                self.posteriors_0,
                                self.x_with_bias,
                                self.y
                            )[0] - yan.YanClassifier._Q(
                                self.yc,
                                self.params,
                                self.D + 1,
                                self.T,
                                self.N,
                                self.posteriors,
                                self.posteriors_0,
                                self.x_with_bias,
                                self.y
                            )[0]) / h[index])
                grad = numpy.mean(grads, axis=0)
                self.assertTrue(numpy.allclose(grad, self.grad[index],
                                               atol=1e-2))

    def test_grad_a(self):
        for d in range(self.D):
            grads = []
            for i in range(self.trials):
                h = numpy.zeros(self.params.shape)
                h[d] = (i + 1) / self.trials * \
                    numpy.linalg.norm(self.params) * 1e-10
                grads.append((yan.YanClassifier._Q(
                            self.yc,
                            self.params + h,
                            self.D + 1,
                            self.T,
                            self.N,
                            self.posteriors,
                            self.posteriors_0,
                            self.x_with_bias,
                            self.y
                        )[0] - yan.YanClassifier._Q(
                            self.yc,
                            self.params,
                            self.D + 1,
                            self.T,
                            self.N,
                            self.posteriors,
                            self.posteriors_0,
                            self.x_with_bias,
                            self.y
                        )[0]) / h[d])
            grad = numpy.mean(grads, axis=0)
            self.assertTrue(numpy.allclose(grad, self.grad[d],
                                           atol=1e-2))


if __name__ == '__main__':
    unittest.main()
