import pickle
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as Scaler#StandardScaler as Scaler
from collections import defaultdict
from lib import *
from elki_parser import *

def validate(our_cluster_info, elki_cluster_info):
    
    # compare number of clusters
    equal_n_clusters = len(our_cluster_info.keys()) == len(elki_cluster_info.keys())
    print(f"The implementations{' ' if equal_n_clusters else ' do not '}return the same number of clusters.")
    print(f"No. of clusters (our ERiC): {len(our_cluster_info.keys())}")
    print(f"No. of clusters (ELKI ERiC): {len(elki_cluster_info.keys())}\n")
    
    # compare number of clusters with identical lambda    
    our_lambda_dict = defaultdict(list)
    elki_lambda_dict = defaultdict(list)
    
    for k in our_cluster_info.keys():
       our_lambda_dict[our_cluster_info[k]['lambda']].append(our_cluster_info[k])
    
    for k in elki_cluster_info.keys():
       elki_lambda_dict[elki_cluster_info[k]['lambda']].append(elki_cluster_info[k])
       
    our_lambda_amounts_dict = {l : len(v) for l, v in our_lambda_dict.items()}
    elki_lambda_amounts_dict = {l : len(v) for l, v in elki_lambda_dict.items()}
    equal_lambda_amounts = our_lambda_amounts_dict==elki_lambda_amounts_dict
    print(f"The implementations{' ' if equal_lambda_amounts else ' do not '}return the same number of lambdas.") 
    print(f"No. of lambdas (our ERiC): {dict(our_lambda_amounts_dict)}")
    print(f"No. of lambdas (ELKI ERiC): {dict(elki_lambda_amounts_dict)}\n")
    
    # compare cluster sizes in each lambda levels
    cluster_sizes_identical = True
    for l in our_lambda_dict.keys():
        if l not in elki_lambda_dict.keys():
            cluster_sizes_identical = False
            print(f"{l} is not an ELKI lambda value")
        
        else:
            our_cluster_sizes = [len(p['points'][0]) for p in our_lambda_dict[l]].sort()
            elki_cluster_sizes = [len(p['points']) for p in elki_lambda_dict[l]].sort()
            
            if our_cluster_sizes == elki_cluster_sizes:
                print(f"Cluster sizes were identical for lambda={l}")
                
            else:
                print(f"Cluster sizes were not identical for lambda={l}")
                cluster_sizes_identical = False
    
    validated = equal_n_clusters and equal_lambda_amounts and cluster_sizes_identical
    print(f"\nValidation result: The outputs of the algorithms are{' ' if validated else ' not '}identical.") 

def load_and_validate(path_our_eric, path_elki_eric, iterations=5, k = 60, alpha = .85, delta_affine = .3, delta_dist = .3, min_samples = 2):

    # loading our implementation's result
    # clustering
    try:
        # reading
        #models = pickle.load(open(os.path.join(path_our_eric, f'models_{iterations}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'rb'))
        #clusters = pickle.load(open(os.path.join(path_our_eric, f'clusters_{iterations}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'rb'))
        #cluster_info = pickle.load(open(os.path.join(path_our_eric, f'cluster_info_{iterations}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.p'), 'rb'))
        df = pd.read_csv("sample_dataset/wages.csv")
        df = df[["AGE", "EDUCATION", "EXPERIENCE", "WAGE"]]
        df.columns = ["A", "YE", "YW", "W"]
        print(df.head())
        print("Samples:", len(df))
        D = df.to_numpy(dtype=np.float64)
        D = Scaler().fit_transform(D)
        point_info, partitions = make_partitions(D, k=k)
        models, clusters = cluster_partitions(D, partitions, point_info, delta_affine=delta_affine, delta_dist=delta_dist, min_samples=min_samples)
        cluster_info = compute_cluster_list(clusters, D)
        cluster_info = build_hierarchy(cluster_info, delta_affine=delta_affine, delta_dist=delta_dist)
    except:
        raise Exception("Loading the clustering failed")
                
    # loading the results of elki eric
    lines = read_file(os.path.join(path_elki_eric, 'elki_eric_output.txt'))#f'elki_output_{iterations}_{k}_{alpha}_{delta_affine}_{delta_dist}_{min_samples}.txt'))
    cluster_info_elki = parse_file(lines)
    validate(cluster_info, cluster_info_elki)
    
load_and_validate("pickles", "")
    