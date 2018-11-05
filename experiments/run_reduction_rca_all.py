from utils import get_string_parser, load_dataset
import matplotlib.pyplot as plt
from clustering import em_cluster, cluster, util
from ca import rca, common
from clustering import util


def get_lambdas(dataset):
    model = rca.train(dataset)

    return rca.get_eigen_values(model)


def plt_all(y1, y2, y3, y4):

    _, (ax1, ax2) = plt.subplots(2, 1)
    x_range_ball = range(2, len(y1) + 2)
    x_range_fire = range(2, len(y3) + 2)
    # explained variance

    ax1.plot(x_range_ball, y1, label="Min Error")
    ax1.plot(x_range_ball, y2, label="Mean Error")

    ax1.set_title("Basketball Representation by RP")
    ax1.set_ylabel("RMS Reconstruction Error")
    ax1.legend()

    ax2.plot(x_range_fire, y3, label="Min Error")
    ax2.plot(x_range_fire, y4, label="Mean Error")
    ax2.set_title("Fire Representation by RP")
    ax2.legend()

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

    all_models_b = []
    min_errors_b = []
    mean_errors_b = []
    for k in range(2, bball_data.shape[1] + 1):
        best_model, min_error, mean_error = rca.single_train(bball_data, k)
        all_models_b.append(best_model)
        min_errors_b.append(min_error)
        mean_errors_b.append(mean_error)

    fire_data = load_dataset("fire")

    all_models_f = []
    min_errors_f = []
    mean_errors_f = []
    for k in range(2, fire_data.shape[1] + 1):
        best_model, min_error, mean_error = rca.single_train(fire_data, k)
        all_models_f.append(best_model)
        min_errors_f.append(min_error)
        mean_errors_f.append(mean_error)

    plt_all(min_errors_b, mean_errors_b, min_errors_f, mean_errors_f)

    best_basketball_model = int(input("choose basketball model"))
    best_fire_index = int(input("choose fire model"))
    # all_fire = []
    # fire_data = load_dataset("fire")
    #
    # for i in range(10):
    #     all_fire.append(get_lambdas(fire_data))
    #
    # # representational strength
    # bball_errors = run_single_experiment(bball_data, rca)
    # fire_errors = run_single_experiment(fire_data, rca)
    #
    # plt_all(all_bball, all_fire)
    #
    # # train components based on previous results

    bball_model = all_models_b[best_basketball_model]
    temp_x, temp_y = util.get_x_y(bball_data)
    bball_x, bball_y = common.single_transform(bball_model, temp_x), temp_y

    fire_model = all_models_f[best_fire_index]
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