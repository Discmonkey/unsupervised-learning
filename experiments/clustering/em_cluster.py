import sklearn.mixture as mixture
from experiments.clustering.util import get_x_y
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def compute_gaussian_mixture(dataset, k):
    gm = mixture.GaussianMixture(n_components=k)
    X, y = get_x_y(dataset)

    gm.fit(X)

    return gm, gm.bic(X), gm.score(X)


def compute_gaussian_mixture_from_x(x, k):
    gm = mixture.GaussianMixture(n_components=k)

    gm.fit(x)

    return gm, gm.bic(x), gm.score(x)

def get_assignments(model, X):
    return model.predict(X)


def create_dataframe_from_assignments(cluster_assignments, labels):
    return pd.DataFrame(np.concatenate((labels, cluster_assignments.reshape(labels.shape)), axis=1),
                        columns=['LABELS', 'CLUSTER_ASSIGNMENTS'])


def calculate_mutual_info(cluster_assignment_dataframe):
    return normalized_mutual_info_score(cluster_assignment_dataframe.LABELS.astype(np.int32).tolist(),
                                        cluster_assignment_dataframe.CLUSTER_ASSIGNMENTS.astype(np.int32).tolist())


def cluster_after_reduction(reducer, x, k):
    fit = reducer.fit_transform(x)

    return compute_gaussian_mixture(fit, k)[0]
