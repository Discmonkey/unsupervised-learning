from scipy import cluster


def k_means_cluster(dataframe, dist_func):
    return cluster.vq.kmeans(dataframe, 30)
