from utils import get_string_parser, load_dataset
import matplotlib.pyplot as plt
from clustering.em_cluster import compute_gaussian_mixture
from ca import pca
from clustering import util


def get_lambdas(dataset):
    model = pca.train_pca(dataset)

    return pca.get_eigen_values(model)


def plt_purity(y1, y2):

    _, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(y1)
    ax1.set_title("Basketball Representation")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Explained Variance Ratio by Component")

    ax2.plot(y2)
    ax2.set_title("Fire ")
    ax2.set_title("Fire Representation")
    ax2.set_xlabel("Component")

    plt.show()


if __name__ == '__main__':

    data = load_dataset("basketball")

    bball_values = get_lambdas(data)

    data2 = load_dataset("fire")

    fire_values = get_lambdas(data2)

    plt_purity(bball_values, fire_values)
