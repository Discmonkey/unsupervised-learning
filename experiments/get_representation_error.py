from experiments.ca import common, ica, pca
from experiments.clustering.util import get_x_y
from experiments.utils import load_dataset
import matplotlib.pyplot as plt


def run_single_experiment(dataset, some_ca, limit=None):
    errors = []
    x, _ = get_x_y(dataset)

    stop = limit if limit is not None else dataset.values.shape[1] - 2

    for i in range(1, stop):
        model = some_ca.train(dataset, i)

        errors.append(common.get_difference(x, common.transform(model, x)))

    return errors


def plt_purity(y1, y2):

    _, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(range(len(y1)), y1)
    ax1.set_title("Basketball Representation Error")
    ax1.set_xlabel("Num Components")
    ax1.set_ylabel("RMS Average Error")

    ax2.plot(range(len(y2)), y2)
    ax2.set_title("Fire Representation Error")
    ax2.set_xlabel("Num Components")

    plt.show()


if __name__ == '__main__':
    fire = load_dataset('fire')
    bball = load_dataset('bball')

    bball_error = run_single_experiment(bball, ica)
    fire_error = run_single_experiment(fire, ica)
    plt_purity(bball_error, fire_error)





