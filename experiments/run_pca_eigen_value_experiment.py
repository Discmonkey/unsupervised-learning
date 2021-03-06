from utils import get_string_parser, load_dataset
import matplotlib.pyplot as plt
from clustering.em_cluster import compute_gaussian_mixture
from ca import pca, common
from clustering import util


def get_model(dataset):
    return pca.train(dataset)


def get_lambdas(model):

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


def run_single_experiment(dataset, some_ca, limit=None):
    errors = []
    x, _ = get_x_y(dataset)

    stop = limit if limit is not None else dataset.values.shape[1] - 2

    for i in range(1, stop):
        model = some_ca.train(dataset, i)

        errors.append(common.get_difference(x, common.transform(model, x)))

    return errors


if __name__ == '__main__':

    # eigen values
    data = load_dataset("basketball")
    model = get_lambdas(data)
    bball_values = get_lambdas(model)

    data2 = load_dataset("fire")
    model2 = get_model(data2)
    fire_values = get_lambdas(model2)

    plt_purity(bball_values, fire_values)

    # representation
