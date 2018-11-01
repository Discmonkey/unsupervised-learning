from utils import get_string_parser, load_dataset
import matplotlib.pyplot as plt
from clustering.em_cluster import compute_gaussian_mixture
from clustering import util


def calculate_mutual_info(dataset, dataset_name, start=2, end=600):
    scores = []
    for i in range(start, end + 1):
        print i
        _, score = compute_gaussian_mixture(dataset, i)

        scores.append(score)

    return scores


def plt_purity(x1, x2, y1, y2):

    _, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(x1, y1)
    ax1.set_title("Basketball Mutual Info")
    ax1.set_xlabel("Num Components")
    ax1.set_ylabel("BIC Score")

    ax2.plot(x2, y2)
    ax2.set_title("BIC Score")
    ax2.set_xlabel("Num Components")

    plt.show()


if __name__ == '__main__':

    data = load_dataset("basketball")

    bball_start, bball_end = 2, 120
    basketball_results = calculate_mutual_info(data, "Basketball", start=bball_start, end=bball_end)

    data2 = load_dataset("fire")

    fire_start, fire_end = 2, 48
    fire_results = calculate_mutual_info(data2, "Fire", start=fire_start, end=fire_end)

    plt_purity(range(bball_start, bball_end + 1), range(fire_start, fire_end + 1), basketball_results, fire_results)
