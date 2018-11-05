from sklearn.decomposition import PCA
from experiments.clustering.util import get_x_y
from scipy.stats import kurtosis


def train(dataframe, num_components=None):
    x, _ = get_x_y(dataframe)
    p = PCA(n_components=num_components)
    p.fit(x)

    return p


def get_vectors(trained_model):
    return trained_model.components_


def get_eigen_values(trained_model):
    return trained_model.explained_variance_ratio_
