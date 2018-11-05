from utils import get_string_parser, load_dataset
import matplotlib.pyplot as plt
from clustering import em_cluster, cluster, util
from ca import ica, common
from clustering import util
from scipy.stats import kurtosis
import numpy as np


def get_kurtosis(dataset):
    model = ica.train(dataset)
    x, _ = util.get_x_y(dataset)
    scores = []

    for i in range(0, len(model.components_)):
        combined = np.matmul(x, model.components_[i])
        scores.append(abs(float(kurtosis(combined))))

    return scores



def plt_all(y1, y2, y3, y4):

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    # explained variance
    ax1.plot(y1)
    ax1.set_title("Basketball ICA")
    ax1.set_ylabel("Kurtosis")

    ax2.plot(y2)
    ax2.set_title("Fire ICA")

    # representational strength
    ax3.plot(y3)
    ax3.set_title("Basketball Representation Error")
    ax3.set_xlabel("Num Components")
    ax3.set_ylabel("RMS Average Error")

    ax4.plot(y4)
    ax4.set_title("Fire Representation Error")
    ax4.set_xlabel("Num Components")

    print "about to show"
    plt.show()


def run_single_experiment(dataset, some_ca, limit=None):
    errors = []
    x, _ = util.get_x_y(dataset)

    stop = limit if limit is not None else dataset.values.shape[1] - 2

    for i in range(1, stop):
        model = some_ca.train(dataset, i)

        errors.append(common.get_difference(x, common.transform(model, x)))

    return errors


def plt_total_representation(y1, y2):

    y1 = [sum(y1[0:i]) for i in range(len(y1))]
    y2 = [sum(y2[0:i]) for i in range(len(y2))]
    _, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(y1)
    ax1.set_title("Basketball Representation")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Total Explained Variance by Number Components Included")

    ax2.plot(y2)
    ax2.set_title("Fire ")
    ax2.set_title("Fire Representation")
    ax2.set_xlabel("Component")

    print "about to show 2"
    plt.show()


def plt_mutual_info(bball_kmean, bball_em, fire_kmean, fire_em):
    x_axis_bball = list(range(2, len(bball_kmean) + 2))
    x_axis_fire = list(range(2, len(fire_kmean) + 2))
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_axis_bball, bball_kmean, label="Kmean")
    ax1.plot(x_axis_bball, bball_em, label="EM")
    ax1.set_title("Basketball Clustering Results")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Mutual Info")
    ax1.legend()

    ax2.plot(x_axis_fire, fire_kmean, label="Kmean")
    ax2.plot(x_axis_fire, fire_em, label="EM")
    ax2.set_title("Fire Clustering Results")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Mutual Info")
    ax2.legend()

    plt.show()


def calculate_mutual_info_kmean(x, y, start=2, end=600):

    def get_purity_error(cluster_tuple):

        # now i'm really being a monster

        centroids, score = cluster_tuple

        classifications = cluster.classify_into_clusters(centroids, x)

        combined_df = cluster.create_dataframe_from_assignments(classifications, y)

        return cluster.calculate_mutual_info(combined_df)

    distortions = cluster.gen_scores(x, start, end, get_purity_error)

    return distortions


def calculate_mutual_info(x, y, start=2, end=600):
    scores = []

    for k in range(start, end + 1):
        print k
        model, _, _ = em_cluster.compute_gaussian_mixture_from_x(x, k)

        predictions = model.predict(x)

        combined = em_cluster.create_dataframe_from_assignments(predictions, y)

        scores.append(em_cluster.calculate_mutual_info(combined))

    return scores


if __name__ == '__main__':

    # explained variance
    bball_data = load_dataset("basketball")

    bball_values = get_kurtosis(bball_data)

    fire_data = load_dataset("fire")

    fire_values = get_kurtosis(fire_data)

    # representational strength
    bball_errors = run_single_experiment(bball_data, ica)
    fire_errors = run_single_experiment(fire_data, ica)

    plt_all(bball_values, fire_values, bball_errors, fire_errors)

    # train components based on previous results
    num_bball_features = int(input("Choose number bball features"))
    bball_model = ica.train(bball_data, num_components=num_bball_features)
    temp_x, temp_y = util.get_x_y(bball_data)
    bball_x, bball_y = common.single_transform(bball_model, temp_x), temp_y

    num_fire_features = int(input("Choose number fire features"))
    fire_model = ica.train(fire_data, num_components=num_fire_features)
    temp_x, temp_y = util.get_x_y(fire_data)
    fire_x, fire_y = common.single_transform(fire_model, temp_x), temp_y

    # mutual info
    bball_start, bball_end = 2, 100
    basketball_mutual_info_cluster = calculate_mutual_info_kmean(bball_x, bball_y, start=bball_start, end=bball_end)
    basketball_mutual_info_em = calculate_mutual_info(bball_x, bball_y, start=bball_start, end=bball_end)

    fire_start, fire_end = 2, 50
    fire_mutual_info_cluster = calculate_mutual_info_kmean(fire_x, fire_y, start=fire_start, end=fire_end)
    fire_mutual_info_em = calculate_mutual_info(fire_x, fire_y, start=fire_start, end=fire_end)

    plt_mutual_info(basketball_mutual_info_cluster, basketball_mutual_info_em,
                    fire_mutual_info_cluster, fire_mutual_info_em)