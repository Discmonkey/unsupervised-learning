from datasets.Fire import Fire
from datasets.Basketball import BasketBall
from ca import pca, ica, nmf, rca
from ca import common
from clustering import cluster, em_cluster, util
import utils
import numpy as np
import pandas as pd


def concat(x, y):
    together = np.concatenate((x, y), axis=1)

    return pd.DataFrame(together)


if __name__ == '__main__':
    print "creating fire datasets"
    f = utils.load_dataset("fire")
    fx, fy = util.get_x_y(f)
    #
    # pca
    pca_components = 10
    pca_model = pca.train(f, num_components=pca_components)
    new_x = pca_model.fit_transform(fx)
    new_df = concat(new_x, fy)

    Fire.save_(new_df, "pca")

    # ica
    ica_model = ica.train(f, num_components=10)
    new_x = ica_model.fit_transform(fx)
    new_df = concat(new_x, fy)
    Fire.save_(new_df, "ica")

    # rca
    best_model = None
    best_score = 100000
    for i in range(100):
        model = rca.train(f, 9)
        score = rca.get_difference(fx, rca.transform(model, fx))

        if score < best_score:
            best_score = score
            best_model = model

    new_x = best_model.fit_transform(fx)
    new_df = concat(new_x, fy)
    Fire.save_(new_df, "rca")

    # nmf
    nmf_model = nmf.train_nmf(f, 10)
    new_x = nmf_model.fit_transform(fx + .5)
    new_df = concat(new_x, fy)
    Fire.save_(new_df, "nmf")

    # clustering
    centroids, _ = cluster.k_means_cluster(fx, 10)
    new_x = cluster.get_cluster_dists(centroids, fx)
    new_x = (new_x - new_x.min()) / (new_x.max() - new_x.min())
    new_df = concat(new_x, fy)
    Fire.save_(new_df, "kmeans")

    # gaussian mixing
    model, _, _ = em_cluster.compute_gaussian_mixture(f, 10)
    probs = model.predict_proba(fx)
    new_df = concat(probs, fy)
    Fire.save_(new_df, "em")

    print "creating basketball datasets"

    f = utils.load_dataset("basketball")
    fx, fy = util.get_x_y(f)
    #
    # pca
    pca_components = 6
    pca_model = pca.train(f, num_components=pca_components)
    new_x = pca_model.fit_transform(fx)
    new_df = concat(new_x, fy)

    BasketBall.save_(new_df, "pca")

    # ica
    ica_model = ica.train(f, num_components=3)
    new_x = ica_model.fit_transform(fx)
    new_df = concat(new_x, fy)
    BasketBall.save_(new_df, "ica")

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
    new_df = concat(new_x, fy)
    BasketBall.save_(new_df, "rca")

    # nmf
    nmf_model = nmf.train_nmf(f, 8)
    new_x = nmf_model.fit_transform(fx + .5)
    new_df = concat(new_x, fy)
    BasketBall.save_(new_df, "nmf")

    # clustering
    centroids, _ = cluster.k_means_cluster(fx, 7)
    new_x = cluster.get_cluster_dists(centroids, fx)
    new_x = (new_x - new_x.min()) / (new_x.max() - new_x.min())
    new_df = concat(new_x, fy)
    BasketBall.save_(new_df, "kmeans")

    # gaussian mixing
    model, _, _ = em_cluster.compute_gaussian_mixture(f, 7)
    probs = model.predict_proba(fx)
    new_df = concat(probs, fy)
    BasketBall.save_(new_df, "em")





















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
