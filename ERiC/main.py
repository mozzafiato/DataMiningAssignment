
import numpy as np
import pandas as pd
from lib import *
from sklearn.preprocessing import StandardScaler as Scaler

# data type important ! float 64 is needed for hashing in dbscan alg.
#D = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float64)

df = pd.read_csv("sample_dataset/wages.csv")
df = df[["AGE", "EDUCATION", "EXPERIENCE", "WAGE"]]
df.columns = ["A", "YE", "YW", "W"]
print(df.head())
print("Samples:", len(df))
D = df.to_numpy(dtype=np.float64)
D = Scaler().fit_transform(D)

point_info, partitions = make_partitions(D, k=200)
print("Partitions:")
print(*[len(p) for p in partitions.values()])

models, clusters = cluster_partitions(D, partitions, point_info, delta_affine=1.5, delta_dist=.5, min_samples=2)
#print(models)
print(clusters.keys())
print(*[len(c) for c in clusters.values()])
#print(clusters[2])
cluster_info = compute_cluster_list(clusters, D)
#print(cluster_info)

hierarchy = build_hierarchy(cluster_info, delta_affine=1.5, delta_dist=.5)

