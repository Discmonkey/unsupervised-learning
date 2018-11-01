from sklearn.decomposition import PCA
from experiments.clustering.util import get_x_y


def train_pca(dataframe):
    x, _ = get_x_y(dataframe)
    p = PCA()
    p.fit(x)

    return p


def get_vectors(trained_model):
    return trained_model.components_


def get_eigen_values(trained_model):
    return trained_model.explained_variance_
