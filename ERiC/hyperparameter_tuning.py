from auxiliarymethods import datasets as dp
from auxiliarymethods.reader import tud_to_networkx
import auxiliarymethods.auxiliary_methods as aux
import os
import numpy as np
from lib import *
import pickle

pickle_path = 'pickles'
# utility functions
def load_csv(path):
    return np.loadtxt(path, delimiter=";")

# hyper parameter
iterations = 5 # weierfeiler-lehman iterations
alpha = .85
min_samples = 2

base_path = os.path.join("kernels", "without_labels")
ds_name = "IMDB-BINARY"

#Gram Matrix for the Weisfeiler-Lehman subtree kernel
try:
    gram = load_csv(os.path.join(base_path,f"{ds_name}_gram_matrix_wl{iterations}.csv"))
except:
    ds_name = "IMDB-BINARY"
    classes = dp.get_dataset(ds_name)
    G = tud_to_networkx(ds_name)
    print(f"Number of graphs in data set is {len(G)}")
    print(f"Number of classes {len(set(classes.tolist()))}")
    gram = load_csv(os.path.join(base_path,f"{ds_name}_gram_matrix_wl{iterations}.csv"))
finally:
    gram = aux.normalize_gram_matrix(gram)


for k in (50, 250, 500):
    for delta_affine in (.2, .5, .7):
        for delta_dist in (.5, 1, 3):
            if not os.path.exists(os.path.join(pickle_path, f'hierarchy_{iterations}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p')):
                print(k, delta_affine, delta_dist)

                point_info = pickle.load(open(os.path.join(pickle_path, f'point_info_{iterations}_{k}_{alpha}.p'), 'rb'))
                partitions = pickle.load(open(os.path.join(pickle_path, f'partitions_{iterations}_{k}_{alpha}.p'), 'rb'))
                # point_info, partitions = make_partitions(gram, k, alpha)
                # pickle.dump(point_info, open(os.path.join(pickle_path, f'point_info_{iterations}_{k}_{alpha}.p'), 'wb'))
                # pickle.dump(partitions, open(os.path.join(pickle_path, f'partitions_{iterations}_{k}_{alpha}.p'), 'wb'))
                models, clusters = cluster_partitions(gram, partitions, point_info, delta_affine, delta_dist, min_samples)
                cluster_info = compute_cluster_list(clusters, gram)
                # writing
                pickle.dump(models, open(os.path.join(pickle_path, f'models_{iterations}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'wb'))
                pickle.dump(clusters, open(os.path.join(pickle_path, f'clusters_{iterations}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'wb'))
                pickle.dump(cluster_info, open(os.path.join(pickle_path, f'cluster_info_{iterations}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'wb'))
                hierarchy = build_hierarchy(cluster_info, delta_affine, delta_dist)
                # writing
                pickle.dump(hierarchy, open(os.path.join(pickle_path, f'hierarchy_{iterations}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'wb'))



