from sklearn.decomposition import PCA
from experiments.clustering.util import get_x_y
from random import randint
from sklearn.random_projection import GaussianRandomProjection
import numpy as np


def train(dataframe, num_components=None):
    x, _ = get_x_y(dataframe)

    p = GaussianRandomProjection(n_components=num_components if num_components is not None else x.shape[1])

    p.fit(x)

    return p


def single_train(dataframe, k, num_sessions=100):
    errors = []
    models = []
    x, y = get_x_y(dataframe)

    for _ in range(num_sessions):
        model = train(dataframe, k)

        errors.append(get_difference(x, transform(model, x)))
        models.append(model)

    best_model = models[int(np.argmin(errors))]
    mean_error = np.mean(errors)
    min_error = np.min(errors)

    return best_model, min_error, mean_error


def get_vectors(trained_model):
    return trained_model.components_


def get_eigen_values(trained_model):
    return trained_model.explained_variance_ratio_


def get_difference(old_representation, new_representation):
    n = len(old_representation)

    dif = old_representation - new_representation

    square_sum = np.sum(dif ** 2, axis=1)

    root = np.sqrt(square_sum)

    return np.sum(root) / n


def transform(trained_model, X):
    fit = trained_model.fit_transform(X)

    return np.matmul(np.linalg.pinv(trained_model.components_), fit.T).T