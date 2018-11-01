import sklearn.mixture as mixture
from experiments.clustering.util import get_x_y


def compute_gaussian_mixture(dataset, k):
    gm = mixture.GaussianMixture(n_components=k)
    X, y = get_x_y(dataset)

    gm.fit(X)

    return gm, gm.bic(X)



