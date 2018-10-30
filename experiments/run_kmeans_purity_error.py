import clustering.util as util
import clustering.cluster as cluster
from top_level_file import base
from os import path
from utils import get_string_parser, load_dataset
import matplotlib.pyplot as plt


def purity_list(dataset, dataset_name, start=2, end=30):
    x, y = util.get_x_y(dataset)

    def get_purity_error(cluster_tuple):

        # now i'm really being a monster


        centroids, score = cluster_tuple

        classifications = cluster.classify_into_clusters(centroids, x)

        combined_df = cluster.create_dataframe_from_assignments(classifications, y)

        return cluster.calculate_purity(combined_df)

    distortions = cluster.gen_scores(x, start, end, get_purity_error)

    plt_purity(range(start, end + 1), distortions, dataset_name)

    return distortions


def plt_purity(range, distortions, dataset_name):
    plt.plot(range, distortions)
    plt.title("Cluster Purity for {} Dataset By K".format(dataset_name.capitalize()))
    plt.xlabel("Num Clusters (K)")
    plt.ylabel("Squared Error")

    plt.show()


if __name__ == '__main__':
    args = get_string_parser("dataset")

    dataset_name = args.dataset
    data = load_dataset(dataset_name)

    purity_list(data, dataset_name)