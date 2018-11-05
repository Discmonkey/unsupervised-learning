import clustering.util as util
import clustering.em_cluster as cluster
from top_level_file import base
from os import path
from utils import get_string_parser, load_dataset
import matplotlib.pyplot as plt


def calculate_mutual_info(dataset, dataset_name, start=2, end=600):
    x, y = util.get_x_y(dataset)

    scores = []

    for k in range(start, end + 1):
        print k
        model, _, _ = cluster.compute_gaussian_mixture(dataset, k)

        predictions = model.predict(x)

        combined = cluster.create_dataframe_from_assignments(predictions, y)

        scores.append(cluster.calculate_mutual_info(combined))

    return scores


def plt_purity(x1, x2, y1, y2):

    _, (ax1, ax2) = plt.subplots(1, 2)

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

    bball_start, bball_end = 2, 100
    basketball_results = calculate_mutual_info(data, "Basketball", start=bball_start, end=bball_end)

    data2 = load_dataset("fire")

    fire_start, fire_end = 2, 60
    fire_results = calculate_mutual_info(data2, "Fire", start=fire_start, end=fire_end)

    plt_purity(range(bball_start, bball_end + 1), range(fire_start, fire_end + 1), basketball_results, fire_results)

