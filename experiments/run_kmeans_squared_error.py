import clustering.util as util
import clustering.cluster as cluster
from top_level_file import base
from os import path
from utils import get_string_parser, load_dataset
import matplotlib.pyplot as plt


def get_clusters(df, k=30):
    x, y = util.get_x_y(df)

    results = cluster.k_means_cluster(x, k)

    classifications = cluster.classify_into_clusters(results[0], x)

    combined_df = cluster.create_dataframe_from_assignments(classifications, y)

    print combined_df


def distortions_list(dataset, dataset_name, start=2, end=60):
    distortions = cluster.gen_scores(dataset, start, end, cluster.distorition_score_func)

    plt_distortions(range(start, end + 1), distortions, dataset_name)

    return distortions


def plt_distortions(range, distortions, dataset_name):
    plt.plot(range, distortions)
    plt.title("Average Error For KMeans {} dataset By K".format(dataset_name))
    plt.xlabel("Num Clusters (K)")
    plt.ylabel("Squared Error")

    plt.show()


if __name__ == '__main__':
    args = get_string_parser("dataset")

    dataset_name = args.dataset
    data = load_dataset(dataset_name)

    distortions_list(data, dataset_name)



