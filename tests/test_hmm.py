from unittest import TestCase

class TestPoisson(TestCase):
    def test_sample_shape(self):
        from hmm import sample_poisson

        self.assertEqual(sample_poisson([3, 3, 3], 1).shape, (3,))
        self.assertEqual(sample_poisson([3], 1).shape, (1,))
        self.assertEqual(sample_poisson([3, 3, 3], 4).shape, (4, 3))
        self.assertEqual(sample_poisson([3], 4).shape, (4, 1))

    def test_poisson_stats(self):
        from hmm import sample_poisson

        lambda_ = 3
        sample = sample_poisson([3], n_samples=4000, random_state=3)
        self.assertLess(abs(sample.mean() - lambda_), 0.1)
        self.assertLess(abs(sample.var() - lambda_), 0.1)

class TestPoissonHMM(TestCase):
    def test_log_likelihood_shape(self):
        from hmm import PoissonHMM
        from numpy import array

        hmm = PoissonHMM(2)
        hmm.lambdas_ = array([[1, 1, 1, 1], [4, 5, 6, 7]])
        sample = array([[0, 2, 4, 7], [4, 5, 4, 5], [1, 5, 5, 1]])
        log_likelihood = hmm._compute_log_likelihood(sample)
        self.assertEqual(log_likelihood.shape, (3, 2))

    def test_fit_known_params(self):
        from hmm import PoissonHMM
        from numpy.random import RandomState

        rand = RandomState(4)

        hmm = PoissonHMM(2)
        hmm.lambdas_ = [[1], [8]]
        hmm.startprob_ = [0.5] * 2
        hmm.transmat_ = [[0.8, 0.2],
                         [0.2, 0.8]]
        x, z = hmm.sample(100, rand)

        model = PoissonHMM(2, params="", init_params="")
        model.lambdas_ = [[1], [8]]
        model.startprob_ = [0.5] * 2
        model.transmat_ = [[0.8, 0.2],
                           [0.2, 0.8]]
        z_p = hmm.predict(x)

if __name__ == '__main__':
    import unittest
    unittest.main()
