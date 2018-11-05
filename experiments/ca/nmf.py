from sklearn.decomposition import NMF
from experiments.clustering.util import get_x_y


def train_nmf(dataframe, num):
    x, _ = get_x_y(dataframe)
    x += .5
    p = NMF(n_components=num)
    p.fit(x)

    return p


def get_vectors(trained_model):
    return trained_model.components_


def get_reconstruction_error(trained_model):
    return trained_model.reconstruction_err_


def get_eigen_values(trained_model):
    return trained_model.explained_variance_ratio_
