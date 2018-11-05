from datasets.Fire import Fire
from datasets.Basketball import BasketBall
from ca import pca, ica, nmf, rca
from ca import common
from clustering import cluster, em_cluster, util
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def concat(x, y):
    together = np.concatenate((x, y), axis=1)

    return pd.DataFrame(together)


def get_kmeans_labels(x):
    results = cluster.k_means_cluster(x, 10)

    classifications = cluster.classify_into_clusters(results[0], x)

    return classifications


def get_em_labels(x):
    model, _, _ = em_cluster.compute_gaussian_mixture_from_x(x, 10)

    return model.predict(x)



if __name__ == '__main__':

    print "creating fire datasets"
    f = utils.load_dataset("fire")
    fx, fy = util.get_x_y(f)

    fire_display_sample = np.random.choice(fx.shape[0], 3000, replace=False)
    # generate axes to project into
    pca_model = pca.train(f, num_components=2)
    grid_axes = pca_model.fit_transform(fx)

    def show(cluster_assignments, title):
        plt.scatter(grid_axes[fire_display_sample, 0],
                    grid_axes[fire_display_sample, 1],
                    c=cluster_assignments[fire_display_sample], s=50, cmap='viridis')

        plt.title(title)
        plt.show()

    default_cluster= get_kmeans_labels(fx)

    show(default_cluster, "cluster default")

    default_em = get_em_labels(fx)

    show(default_em, "em default")


    # plt.scatter(grid_axes[fire_display_sample, 0],
    #             grid_axes[fire_display_sample, 1],
    #             c=default_em[fire_display_sample], s=50, cmap='viridis')
    # plt.show()
    #
    # pca
    pca_components = 10
    pca_model = pca.train(f, num_components=pca_components)
    new_x = pca_model.fit_transform(fx)
    default_cluster = get_kmeans_labels(new_x)
    default_em = get_em_labels(new_x)
    show(default_cluster, "pca kmeans")
    show(default_em, "pca em")

    # ica
    ica_model = ica.train(f, num_components=10)
    new_x = ica_model.fit_transform(fx)
    default_cluster = get_kmeans_labels(new_x)
    default_em = get_em_labels(new_x)

    show(default_cluster, "ica kmeans")
    show(default_em, "ica em")
    #
    # # rca
    best_model = None
    best_score = 100000
    for i in range(100):
        model = rca.train(f, 9)
        score = rca.get_difference(fx, rca.transform(model, fx))

        if score < best_score:
            best_score = score
            best_model = model

    new_x = best_model.fit_transform(fx)
    default_cluster = get_kmeans_labels(new_x)
    default_em = get_em_labels(new_x)

    show(default_cluster, "ica kmeans")
    show(default_em, "ica em")
    #
    # # nmf
    nmf_model = nmf.train_nmf(f, 10)
    new_x = nmf_model.fit_transform(fx + .5)
    default_cluster = get_kmeans_labels(new_x)
    default_em = get_em_labels(new_x)

    show(default_cluster, "ica kmeans")
    show(default_em, "ica em")
    #

    #
    # print "creating basketball datasets"
    #
    f = utils.load_dataset("basketball")
    fx, fy = util.get_x_y(f)

    fire_display_sample = np.random.choice(fx.shape[0], 3000, replace=False)
    # generate axes to project into
    pca_model = pca.train(f, num_components=2)
    grid_axes = pca_model.fit_transform(fx)


    def show(cluster_assignments, title):
        plt.scatter(grid_axes[fire_display_sample, 0],
                    grid_axes[fire_display_sample, 1],
                    c=cluster_assignments[fire_display_sample], s=50, cmap='viridis')

        plt.title(title)
        plt.show()


    default_cluster = get_kmeans_labels(fx)
    default_em = get_em_labels(fx)
    show(default_cluster, "pca kmeans")
    show(default_em, "pca em")
    #
    # pca
    pca_components = 6
    pca_model = pca.train(f, num_components=pca_components)
    new_x = pca_model.fit_transform(fx)
    default_cluster = get_kmeans_labels(new_x)
    default_em = get_em_labels(new_x)
    show(default_cluster, "pca kmeans")
    show(default_em, "pca em")

    # ica
    ica_model = ica.train(f, num_components=3)
    new_x = ica_model.fit_transform(fx)
    default_cluster = get_kmeans_labels(new_x)
    default_em = get_em_labels(new_x)
    show(default_cluster, "ica kmeans")
    show(default_em, "ica em")

    # rca
    best_model = None
    best_score = 100000
    for i in range(100):
        model = rca.train(f, 4)
        score = rca.get_difference(fx, rca.transform(model, fx))

        if score < best_score:
            best_score = score
            best_model = model

    new_x = best_model.fit_transform(fx)
    default_cluster = get_kmeans_labels(new_x)
    default_em = get_em_labels(new_x)
    show(default_cluster, "rca kmeans")
    show(default_em, "rca em")

    # nmf
    nmf_model = nmf.train_nmf(f, 8)
    new_x = nmf_model.fit_transform(fx + .5)
    default_cluster = get_kmeans_labels(new_x)
    default_em = get_em_labels(new_x)
    show(default_cluster, "nmf kmeans")
    show(default_em, "nmf em")





















    #
    # b = utils.load_dataset("bball")
    # bx, by = util.get_x_y(b.df_processed)
    #
    # print "Performing PCA"
    # num_basketball, num_fire = 6, 10
    #
    # basketball_model = pca.train(b, num_components=num_basketball)
    #
    #
    # def ball_transform(data):
    #     return basketball_model.fit_transform(data)
    #
    #
    # b_data = b.transform(ball_transform)
    # b.save_(b_data, "pca_ball")
    #
    # fire_model = pca.train(f, num_components=num_fire)
    #
    # reduced = fire_model.fit_transform(fx)
