import numpy as np
import pandas as pd
from lib import *


# data type important ! float 64 is needed for hashing in dbscan alg.
#D = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float64)

df = pd.read_csv("sample_dataset/wages.csv")
df = df[["AGE", "EDUCATION", "EXPERIENCE", "WAGE"]]
df.columns = ["A", "YE", "YW", "W"]
print(df.head())
D = df.to_numpy(dtype=np.float64)

point_info, partitions = make_partitions(D, k=120)
print("Info per point:")
#print(point_info)
print("Partitions:")
print(*[len(p) for p in partitions.values()])

models, clusters = cluster_partitions(D, partitions, point_info, 1.5, .5, 2)
print(models)
print(clusters)

