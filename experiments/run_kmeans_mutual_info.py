import clustering.util as util
import clustering.cluster as cluster
from top_level_file import base
from os import path
from utils import get_string_parser, load_dataset
import matplotlib.pyplot as plt


def calculate_mutual_info(dataset, dataset_name, start=2, end=600):
    x, y = util.get_x_y(dataset)

    def get_purity_error(cluster_tuple):

        # now i'm really being a monster

        centroids, score = cluster_tuple

        classifications = cluster.classify_into_clusters(centroids, x)

        combined_df = cluster.create_dataframe_from_assignments(classifications, y)

        return cluster.calculate_mutual_info(combined_df)

    distortions = cluster.gen_scores(x, start, end, get_purity_error)

    return distortions


def plt_purity(x1, x2, y1, y2):

    _, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.plot(x1, y1)
    ax1.set_title("Basketball Mutual Info")
    ax1.set_xlabel("Num Clusters (K)")
    ax1.set_ylabel("Mutual Info")

    ax2.plot(x2, y2)
    ax2.set_title("Fire Mutual Info")
    ax2.set_xlabel("Num Clusters (K)")

    plt.show()


if __name__ == '__main__':

    data = load_dataset("basketball")

    bball_start, bball_end = 2, 480
    basketball_results = calculate_mutual_info(data, "Basketball", start=bball_start, end=bball_end)

    data2 = load_dataset("fire")

    fire_start, fire_end = 2, 120
    fire_results = calculate_mutual_info(data2, "Fire", start=fire_start, end=fire_end)

    plt_purity(range(bball_start, bball_end + 1), range(fire_start, fire_end + 1), basketball_results, fire_results)

