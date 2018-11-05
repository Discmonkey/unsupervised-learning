from sklearn.decomposition import FastICA
from experiments.clustering.util import get_x_y
import numpy as np


def train(dataframe, num_components=None):
    x, _ = get_x_y(dataframe)
    p = FastICA(n_components=num_components, max_iter=800, whiten=True)
    p.fit(x)

    return p


def get_vectors(trained_model):
    return trained_model.components_


def get_eigen_values(trained_model):
    return trained_model.explained_variance_


