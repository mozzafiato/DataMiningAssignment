from auxiliarymethods import datasets as dp
from auxiliarymethods.reader import tud_to_networkx
import auxiliarymethods.auxiliary_methods as aux
import os
import numpy as np
from lib import *
import pickle
from matplotlib import pyplot as plt
from elki_parser import draw_graph

pickle_path = 'pickles'
# utility functions
def load_csv(path):
    return np.loadtxt(path, delimiter=";")

# hyper parameter
k = 250
delta_affine = .5
delta_dist = .5
alpha = .85
min_samples = 2

base_path = os.path.join("kernels", "without_labels")
ds_name = "IMDB-BINARY"




for kernel in ("graphlet", "shortestpath", "wl1", "wl2", "wl3", "wl4", "wl5"):

    gram = load_csv(os.path.join(base_path,f"{ds_name}_gram_matrix_{kernel}.csv"))
    gram = aux.normalize_gram_matrix(gram)

    if kernel.startswith("wl"):
        kernel = kernel[-1]

    print(np.count_nonzero(np.isinf(gram)))

    print(kernel)

    try:
        # partitioning
        try:
            # reading
            point_info = pickle.load(open(os.path.join(pickle_path, f'point_info_{kernel}_{k}_{alpha}.p'), 'rb'))
            partitions = pickle.load(open(os.path.join(pickle_path, f'partitions_{kernel}_{k}_{alpha}.p'), 'rb'))
        except:
            # *** COMPUTATION ***
            point_info, partitions = make_partitions(gram, k, alpha)
            # writing
            pickle.dump(point_info, open(os.path.join(pickle_path, f'point_info_{kernel}_{k}_{alpha}.p'), 'wb'))
            pickle.dump(partitions, open(os.path.join(pickle_path, f'partitions_{kernel}_{k}_{alpha}.p'), 'wb'))


        # clustering
        try:
            # reading
            models = pickle.load(open(os.path.join(pickle_path, f'models_{kernel}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'rb'))
            clusters = pickle.load(open(os.path.join(pickle_path, f'clusters_{kernel}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'rb'))
            cluster_info = pickle.load(open(os.path.join(pickle_path, f'cluster_info_{kernel}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'rb'))
        except:
            # *** COMPUTATION ***
            models, clusters = cluster_partitions(gram, partitions, point_info, delta_affine, delta_dist, min_samples)
            cluster_info = compute_cluster_list(clusters, gram)
            # writing
            pickle.dump(models, open(os.path.join(pickle_path, f'models_{kernel}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'wb'))
            pickle.dump(clusters, open(os.path.join(pickle_path, f'clusters_{kernel}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'wb'))
            pickle.dump(cluster_info, open(os.path.join(pickle_path, f'cluster_info_{kernel}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'wb'))



        # hierarchy
        try:
            # reading
            hierarchy = pickle.load(open(os.path.join(pickle_path, f'hierarchy_{kernel}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'rb'))
        except:
            # *** COMPUTATION ***
            hierarchy = build_hierarchy(cluster_info, delta_affine, delta_dist, gram.shape[1])
            # writing
            pickle.dump(hierarchy, open(os.path.join(pickle_path, f'hierarchy_{kernel}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'wb'))

        fig = plt.figure((13, 17))
        draw_graph(hierarchy, )
        fig.savefig(f'plot_{kernel}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.png')




    except: pass



