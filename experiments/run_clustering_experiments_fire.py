import clustering.util as util
import clustering.cluster as cluster
from top_level_file import base
from os import path


DATASET = path.join(base, '..', "datasets", "cache", "fire_all.csv")

df = util.load_data_frame(DATASET)

x, y = util.get_x_y(df)


results = cluster.k_means_cluster(x, 30)

classifications = cluster.classify_into_clusters(results[0], x)

print classifications

