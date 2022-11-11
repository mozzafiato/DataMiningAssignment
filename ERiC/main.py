import numpy as np
from sklearn.neighbors import NearestNeighbors
from lib import *




###### MAIN ############

D = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
info_point, partitions = make_partitions(D)
print("Info per point:")
print(info_point)
print("Partitions:")
print(partitions)

