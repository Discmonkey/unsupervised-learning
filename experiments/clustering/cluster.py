from scipy import cluster
import numpy as np
import pandas as pd


def k_means_cluster(observations, k):
    # normalize across all features, recommended in the documentation
    whitened = cluster.vq.whiten(observations)
    return cluster.vq.kmeans(whitened, k)


def classify_into_clusters(centroids, observations):
    whiten = cluster.vq.whiten(observations)
    widen = whiten[:, np.newaxis, :]

    scores = np.sum((widen - centroids)**2, axis=2)

    cluster_assignments = np.argmin(scores, axis=1)

    return cluster_assignments


def create_dataframe_from_assignments(cluster_assignments, labels):
    return pd.DataFrame(np.concatenate((labels, cluster_assignments.reshape(labels.shape)), axis=1),
                        columns=['LABELS', 'CLUSTER_ASSIGNMENTS'])


def calculate_purity(cluster_assignment_dataframe):
    cluster_assignment_dataframe['count'] = 1
    # i'm a monster
    total = cluster_assignment_dataframe.groupby(['CLUSTER_ASSIGNMENTS', 'LABELS']).count().reset_index().groupby(
        ['CLUSTER_ASSIGNMENTS']).max()['count'].sum()

    return float(total) / len(cluster_assignment_dataframe)


def distorition_score_func(k_means_tuple):
    return k_means_tuple[1]


def gen_scores(observations, min_clusters=2, max_clusters=20, score_func=None):
    if score_func is None:
        raise ValueError("No score func provided")

    distortions = []
    for i in range(min_clusters, max_clusters + 1):
        print i
        distortions.append(score_func(k_means_cluster(observations, i)))

    return distortions

