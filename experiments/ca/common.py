import numpy as np


def get_difference(old_representation, new_representation):
    n = len(old_representation)

    dif = old_representation - new_representation

    square_sum = np.sum(dif ** 2, axis=1)

    root = np.sqrt(square_sum)

    return np.sum(root) / n


def transform(trained_model, X):
    fit = trained_model.fit_transform(X)

    return trained_model.inverse_transform(fit)


def single_transform(trained_model, x):
    return trained_model.fit_transform(x)
