from scipy import cluster
import numpy as np
import pandas as pd


def k_means_cluster(observations, k):
    # normalize across all features, recommended in the documentation
    whitened = cluster.vq.whiten(observations)
    return cluster.vq.kmeans(whitened, k)


def classify_into_clusters(centroids, observations):
    widen = observations[:, np.newaxis, :]

    scores = np.sum((widen - centroids)**2, axis=2)

    cluster_assignments = np.argmin(scores, axis=1)

    return cluster_assignments


def create_dataframe_from_assignments(cluster_assignments, labels):
    return pd.DataFrame(np.concatenate((labels, cluster_assignments.reshape(labels.shape)), axis=1),
                        columns=['LABELS', 'CLUSTER_ASSIGNMENTS'])


def gen_distortion_list(observations, min_clusters=2, max_clusters=20):
    distortions = []
    for i in range(min_clusters, max_clusters + 1):
        print i
        distortions.append(k_means_cluster(observations, i)[1])

    return distortions
