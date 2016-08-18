import unittest

import numpy

from . import passive_crowd

# Most of the tests here are regression or shape tests. Shape in particular -
# vectorising is hard and error-prone.


class TestLogisticRegression(unittest.TestCase):

    def test_logistic_regression_vector(self):
        """logistic_regression works on vectors of data."""
        a = numpy.ones(10)
        b = 1
        x = numpy.ones(10)
        self.assertTrue(numpy.isclose(
                passive_crowd.logistic_regression(a, b, x),
                0.999983))

    def test_logistic_regression_matrix(self):
        """logistic_regression works on matrices of data."""
        a = numpy.ones(10)
        b = 1
        x = numpy.ones((20, 10)) * numpy.linspace(0, 1, 10)
        self.assertTrue(numpy.allclose(
                passive_crowd.logistic_regression(a, b, x),
                numpy.ones(20) * 0.997527))


class TestAnnotatorModel(unittest.TestCase):

    def test_annotator_model_vector(self):
        """annotator_model works on vectors of data."""
        w = numpy.linspace(0, 1, 10)
        g = -1
        x = numpy.linspace(0, 1, 10)
        y = 1
        z = 1
        self.assertTrue(numpy.isclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                0.92552988))

        z = 0
        self.assertTrue(numpy.isclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                0.0744701169))

    def test_annotator_model_matrix(self):
        """annotator_model works on matrices of data."""
        w = numpy.linspace(0, 1, 10)
        g = -1
        x = numpy.ones((20, 10)) * numpy.linspace(0, 1, 10)
        y = 1
        z = 1
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones(20) * 0.92552988))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (20,))

        z = 0
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones(20) * 0.0744701169))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (20,))

    def test_annotator_model_anno_vector(self):
        """annotator_model works on vectors of annotators and data vectors."""
        w = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 5, axis=0)
        g = -numpy.ones(5)
        x = numpy.linspace(0, 1, 10)
        y = numpy.ones(5)
        z = 1
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones(5) * 0.92552988))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (5,))

        z = 0
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones(5) * 0.0744701169))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (5,))

    def test_annotator_model_masked_anno_vector(self):
        """annotator_model works on masked vectors of annotators."""
        w = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 5, axis=0)
        g = -numpy.ones(5)
        x = numpy.linspace(0, 1, 10)
        y = numpy.ma.MaskedArray(numpy.ones(5), mask=[0, 0, 1, 0, 1])
        z = 1
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones(5) * 0.92552988))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (5,))
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z).mask,
                y.mask))

    def test_annotator_model_anno_vector(self):
        """annotator_model works on vectors of annotators and data matrices."""
        w = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 5, axis=0)
        g = -numpy.ones(5)
        x = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 20, axis=0)
        y = numpy.ones((5, 20))
        z = 1
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones((5, 20)) * 0.92552988))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (5, 20))

        z = 0
        self.assertTrue(numpy.allclose(
                passive_crowd.annotator_model(w, g, x, y, z),
                numpy.ones((5, 20)) * 0.0744701169))
        self.assertEqual(
                passive_crowd.annotator_model(w, g, x, y, z).shape,
                (5, 20))


class TestQ(unittest.TestCase):

    def test_q(self):
        """Q returns correct value and gradients."""
        n_dim = 10
        n_samples = 20
        n_annotators = 5
        posteriors = numpy.linspace(0, 1, n_samples)
        posteriors_0 = 1 - posteriors
        x = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 20, axis=0)
        y = numpy.ones((5, 20))
        a = numpy.ones(10)
        b = 1
        w = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 5, axis=0)
        g = -numpy.ones(5)
        params = passive_crowd.pack(a, b, w, g)
        value, grad = passive_crowd.Q(params, n_dim, n_annotators, n_samples,
                posteriors, posteriors_0, x, y)

        self.assertTrue(numpy.isclose(
                value,
                433.98487891991613))
        self.assertTrue(numpy.allclose(
                grad,
                numpy.array([
                    -0.        ,   5.52808196,  11.05616393,  16.58424589,
                    22.11232786,  27.64040982,  33.16849179,  38.69657375,
                    44.22465572,  49.75273768,  49.75273768,  -0.        ,
                     0.94539974,   1.89079948,   2.83619922,   3.78159896,
                     4.7269987 ,   5.67239844,   6.61779818,   7.56319792,
                     8.50859766,  -0.        ,   0.94539974,   1.89079948,
                     2.83619922,   3.78159896,   4.7269987 ,   5.67239844,
                     6.61779818,   7.56319792,   8.50859766,  -0.        ,
                     0.94539974,   1.89079948,   2.83619922,   3.78159896,
                     4.7269987 ,   5.67239844,   6.61779818,   7.56319792,
                     8.50859766,  -0.        ,   0.94539974,   1.89079948,
                     2.83619922,   3.78159896,   4.7269987 ,   5.67239844,
                     6.61779818,   7.56319792,   8.50859766,  -0.        ,
                     0.94539974,   1.89079948,   2.83619922,   3.78159896,
                     4.7269987 ,   5.67239844,   6.61779818,   7.56319792,
                     8.50859766,   8.50859766,   8.50859766,   8.50859766,
                     8.50859766,   8.50859766])))


class TestEMStep(unittest.TestCase):

    def test_em_step(self):
        """em_step returns correct parameters."""
        n_dim = 10
        n_samples = 20
        n_annotators = 5
        x = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 20, axis=0)
        y = numpy.ones((5, 20))
        a = numpy.ones(10)
        b = 1
        w = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 5, axis=0)
        g = -numpy.ones(5)
        a_, b_, w_, g_ = passive_crowd.em_step(
                n_samples, n_annotators, n_dim, a, b, w, g, x, y)

        self.assertTrue(numpy.allclose(
            a_,
            numpy.array([
                1.        ,  1.3468034 ,  1.6936068 ,  2.04041019,  2.38721359,
                2.73401699,  3.08082039,  3.42762379,  3.77442718,  4.12123058,
            ])))

        self.assertTrue(numpy.isclose(b_, 4.1212305822007558))

        self.assertTrue(numpy.allclose(
            w_,
            numpy.array([[
                0.        ,  0.49182088,  0.98364176,  1.47546264,  1.96728352,    
                2.4591044 ,  2.95092527,  3.44274615,  3.93456703,  4.42638791],   
              [ 0.        ,  0.49182088,  0.98364176,  1.47546264,  1.96728352,    
                2.4591044 ,  2.95092527,  3.44274615,  3.93456703,  4.42638791],   
              [ 0.        ,  0.49182088,  0.98364176,  1.47546264,  1.96728352,    
                2.4591044 ,  2.95092527,  3.44274615,  3.93456703,  4.42638791],   
              [ 0.        ,  0.49182088,  0.98364176,  1.47546264,  1.96728352,    
                2.4591044 ,  2.95092527,  3.44274615,  3.93456703,  4.42638791],   
              [ 0.        ,  0.49182088,  0.98364176,  1.47546264,  1.96728352,    
                2.4591044 ,  2.95092527,  3.44274615,  3.93456703,  4.42638791],
            ])))

        self.assertTrue(numpy.allclose(
            g_,
            numpy.array([
                2.42638791,  2.42638791,  2.42638791,  2.42638791,  2.42638791,
            ])))


class TestTrain(unittest.TestCase):
    
    def test_train(self):
        """train returns correct result."""
        numpy.random.seed(0)
        x = numpy.repeat(numpy.linspace(0, 1, 10).reshape((1, 10)), 20, axis=0)
        y = numpy.ones((5, 20), dtype=bool)
        a, b, w, g = passive_crowd.train(x, y)

        self.assertTrue(numpy.allclose(
            a,
            numpy.array([
                1.76405235, 0.82518343, 1.82879042, 3.51597185, 3.56766286,
                1.1478532, 3.50024572, 2.82382631, 3.29699088, 4.23583445,
            ])))

        self.assertTrue(numpy.isclose(b, 3.9692795228301705))

        self.assertTrue(numpy.allclose(
            w,
            numpy.array([[
                1.45427351,  1.22550216,  1.05060388,  1.83725652,  2.19153205,
                3.81640122,  2.58162832,  3.56431871,  2.8616197 ,  1.62719006],
              [ 0.6536186 ,  1.21670254, -0.03763233,  3.32655366, -0.04530029,
                1.80709024,  1.92641422,  3.99864363,  4.28748953,  3.32534453],
              [ 0.37816252, -0.45403019, -1.11328536,  0.95335451,  1.89137119,
                3.39906845,  3.80491318,  2.64896206,  3.16774169,  2.85524703],
              [-1.42001794, -1.22531615,  2.91268348,  0.93320995,  1.48574188,
                1.15197486,  3.66321462,  1.75278046,  3.63489207,  3.43311984],
              [ 0.3869025 , -0.06059712, -0.28021615,  1.32244183,  2.22916394,
                2.31755731,  3.00372001,  2.51713403,  3.23892298,  3.37941171],
            ])))

        self.assertTrue(numpy.allclose(
            g,
            numpy.array([
                5.18017987,  4.17039711,  4.90379999,  5.3285864 ,  5.05187216,
            ])))


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
        self.y = numpy.random.binomial(1, 0.5, size=(T, N))
        
        self.posteriors = numpy.random.random(size=(N,))
        self.posteriors_0 = 1 - self.posteriors

        self.params = numpy.random.normal(scale=0.5, size=(D + 1 + T * D + T,))

        self.grad = passive_crowd.Q(
                self.params, D, T, N, self.posteriors, self.posteriors_0,
                self.x, self.y)[1]

        self.trials = 10

    # A neater way to do this would be to just loop over all indices in the
    # gradient. I've chosen to split this into each parameter so that the tests
    # will better indicate which gradients are incorrect.

    def test_grad_b(self):
        grads = []
        for i in range(self.trials):
            h = numpy.zeros(self.params.shape)
            h[self.D] = (i + 1) / self.trials \
                    * numpy.linalg.norm(self.params) * 1e-10
            grads.append((passive_crowd.Q(
                        self.params + h,
                        self.D,
                        self.T,
                        self.N,
                        self.posteriors,
                        self.posteriors_0,
                        self.x,
                        self.y
                    )[0] - passive_crowd.Q(
                        self.params,
                        self.D,
                        self.T,
                        self.N,
                        self.posteriors,
                        self.posteriors_0,
                        self.x,
                        self.y
                    )[0]) / h[self.D])
        grad = numpy.mean(grads, axis=0)
        self.assertTrue(numpy.isclose(grad, self.grad[self.D], atol=1e-2))

    def test_grad_g(self):
        for t in range(self.T):
            grads = []
            for i in range(self.trials):
                h = numpy.zeros(self.params.shape)
                h[-self.T+t] = (i + 1) / self.trials \
                        * numpy.linalg.norm(self.params) * 1e-10
                grads.append((passive_crowd.Q(
                            self.params + h,
                            self.D,
                            self.T,
                            self.N,
                            self.posteriors,
                            self.posteriors_0,
                            self.x,
                            self.y
                        )[0] - passive_crowd.Q(
                            self.params,
                            self.D,
                            self.T,
                            self.N,
                            self.posteriors,
                            self.posteriors_0,
                            self.x,
                            self.y
                        )[0]) / h[-self.T+t])
            grad = numpy.mean(grads, axis=0)
            self.assertTrue(numpy.allclose(grad, self.grad[-self.T+t],
                                           atol=1e-2))

    def test_grad_w(self):
        for t in range(self.T):
            for d in range(self.D):
                grads = []
                index = self.D+1+self.D*t+d
                for i in range(self.trials):
                    h = numpy.zeros(self.params.shape)
                    h[index] = (i + 1) / self.trials \
                            * numpy.linalg.norm(self.params) * 1e-10
                    grads.append((passive_crowd.Q(
                                self.params + h,
                                self.D,
                                self.T,
                                self.N,
                                self.posteriors,
                                self.posteriors_0,
                                self.x,
                                self.y
                            )[0] - passive_crowd.Q(
                                self.params,
                                self.D,
                                self.T,
                                self.N,
                                self.posteriors,
                                self.posteriors_0,
                                self.x,
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
                h[d] = (i + 1) / self.trials \
                        * numpy.linalg.norm(self.params) * 1e-10
                grads.append((passive_crowd.Q(
                            self.params + h,
                            self.D,
                            self.T,
                            self.N,
                            self.posteriors,
                            self.posteriors_0,
                            self.x,
                            self.y
                        )[0] - passive_crowd.Q(
                            self.params,
                            self.D,
                            self.T,
                            self.N,
                            self.posteriors,
                            self.posteriors_0,
                            self.x,
                            self.y
                        )[0]) / h[d])
            grad = numpy.mean(grads, axis=0)
            self.assertTrue(numpy.allclose(grad, self.grad[d],
                                           atol=1e-2))


if __name__ == '__main__':
    unittest.main()
