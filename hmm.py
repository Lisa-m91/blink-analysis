from hmmlearn import hmm
from sklearn import cluster
import numpy as np

def sample_poisson(lambda_, n_samples=1, random_state=None):
    from sklearn.utils import check_random_state

    lambda_ = np.asarray(lambda_)
    rng = check_random_state(random_state)
    ndim = lambda_.shape[0]
    sample = rng.poisson(lambda_, (n_samples, ndim))
    if n_samples == 1:
        sample.shape = sample.shape[1:]
    return sample

class PoissonHMM(hmm._BaseHMM):
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="stl", init_params="stl"):
        super().__init__(n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior,
                         algorithm=algorithm,
                         random_state=random_state,
                         n_iter=n_iter, tol=tol, verbose=verbose,
                         params=params, init_params=init_params)

    def _init(self, X, lengths):
        # Sets startprob_ and transmat_ to uniform if not given
        super()._init(X, lengths)

        n_samples, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError("Unexpected number of dimensions, got {} but expected {}"
                             .format(n_features, self.n_features))
        self.n_features = n_features

        if 'l' in self.init_params or not hasattr(self, "lambdas_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            self.lambdas_ = kmeans.cluster_centers_

    def _check(self):
        # Checks startprob_ and transmat_ sum to 1
        super()._check()

        self.lambdas_ = np.asarray(self.lambdas_)
        _, self.n_features = self.lambdas_.shape

    def _compute_log_likelihood(self, X):
        from numpy import log
        # https://onlinecourses.science.psu.edu/stat504/node/31

        # from scipy.misc import factorial
        # Note ignores constant factor (+ log(factorial(X)))

        n_samples = len(X)
        lambdas = self.lambdas_.T[:, None, :] # (n_features, 1, n_components)
        X = X.T[..., None] # (n_features, n_samples, 1)

        return sum(X * log(lambdas) - lambdas * n_samples)

    def _generate_sample_from_state(self, state, random_state=None):
        state = int(state)
        return sample_poisson(self.lambdas_[state], random_state=random_state)

    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()

        # Total posterior for each component over all samples
        stats['post'] = np.zeros(self.n_components)
        # Component means
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statictics(stats, X, framelogprob, posteriors,
                                                  fwdlattice, bwdlattice)
        if 'l' in self.params:
            # Total posterior for normalization in M-step
            stats['post'] += posteriors.sum(axis=0)
            # Component means weighted by posterior
            stats['obs'] += np.dot(posteriors.T, obs)

    def _do_mstep(self, stats):
        # Update startprob and transition matrix
        super()._do_mstep(stats)

        if 'l' in self.params:
            self.lambdas_ = stats['obs'] / stats['post'][:, None]

def normalize(matrix):
    matrix = np.asarray(matrix)
    return matrix / np.sum(matrix, axis=1, keepdims=True)

def trainMultiple(model, *traces):
    lengths = list(map(len, traces))
    data = np.concatenate(traces)
    return model.fit(data, lengths)

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    simple_trans = [[1.0, 0.2, 0.0], # off
                    [0.2, 1.0, 0.1], # on
                    [0.0, 0.0, 1.0]] # bleached
    simple_means = ([0.0], [1.5], [0.0])
    simple_covar = ([0.1], [0.1], [0.1])

    two_state_trans = [[1.0, 0.2, 0.1, 0.1, 0.0], # dye/off
                       [0.2, 1.0, 0.1, 0.1, 0.1], # dye/on
                       [0.0, 0.0, 1.0, 0.2, 0.0], # eos/off
                       [0.0, 0.0, 0.4, 1.0, 0.3], # eos/on
                       [0.0, 0.0, 0.0, 0.0, 1.0]] # bleached
    two_state_means = ([0.0], [1.0], [0.0], [1.0], [0.0])
    two_state_covar = ([0.1], [0.1], [0.1], [0.1], [0.1])

    model = hmm.GaussianHMM(n_components=len(simple_trans))
    model.startprob_ = [0.5, 0.5] + [0.0] * (len(simple_trans) - 2)
    model.transmat_ = normalize(simple_trans)
    model.means_ = simple_means
    model.covars_ = simple_covar
    x, z = model.sample(100)

    remodel = hmm.GaussianHMM(n_components=len(simple_trans), params="mt", init_params="mt")
    remodel.startprob_ = [0.5, 0.5] + [0.0] * (len(simple_trans) - 2)
    remodel.covars_ = simple_covar
    remodel.fit(x)
    z2 = remodel.predict(x)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x)
    plt.show()
