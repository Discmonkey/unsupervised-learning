from utils import get_string_parser, load_dataset
import matplotlib.pyplot as plt
from clustering.em_cluster import compute_gaussian_mixture
from clustering import util


def calculate_mutual_info(dataset, dataset_name, start=2, end=600):
    scores = []
    scores2 = []

    for i in range(start, end + 1):
        print i
        _, score1, score2 = compute_gaussian_mixture(dataset, i)

        scores.append(score1)
        scores2.append(score2)

    return scores, scores2


def plt_purity(x1, x2, y1, y2, y3, y4):

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.plot(x1, y1)
    ax1.set_title("Basketball BIC Score")
    ax1.set_ylabel("BIC Score")

    ax2.plot(x2, y2)
    ax2.set_title("Fire ")
    ax2.set_title("Fire BIC Score")

    ax3.plot(x1, y3)
    ax3.set_title("Basketball Non Penalized Score")
    ax3.set_xlabel("Num Components")
    ax3.set_ylabel("Score")

    ax4.plot(x2, y4)
    ax4.set_title("Fire ")
    ax4.set_title("Fire Non Penalized Score")
    ax4.set_xlabel("Num Components")

    plt.show()


if __name__ == '__main__':

    data = load_dataset("basketball")

    bball_start, bball_end = 2, 20
    basketball_results = calculate_mutual_info(data, "Basketball", start=bball_start, end=bball_end)

    data2 = load_dataset("fire")

    fire_start, fire_end = 2, 3
    fire_results = calculate_mutual_info(data2, "Fire", start=fire_start, end=fire_end)

    plt_purity(range(bball_start, bball_end + 1),
               range(fire_start, fire_end + 1), basketball_results[0], fire_results[0], basketball_results[1], fire_results[1])
