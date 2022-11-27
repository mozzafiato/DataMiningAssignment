from collections import defaultdict


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

    print("Our ERiC structure:")
    for l in our_lambda_dict.keys():
        print("Partition ", l)
        for c in our_cluster_info.keys():
            if l == our_cluster_info[c]['lambda']:
                print("--- cluster", our_cluster_info[c]['index'], " size:", len(our_cluster_info[c]['points']))
                print(sorted(our_cluster_info[c]['points']))

    print("")
    print("ELKI ERic structure")
    for l in elki_lambda_dict.keys():
        print("Partition ", l)
        for c in elki_cluster_info.keys():
            if l == elki_cluster_info[c]['lambda']:

                print("--- cluster",  elki_cluster_info[c]['index'], " size:", len(elki_cluster_info[c]['points']))
                print(sorted(elki_cluster_info[c]['points']))

    print("")
    # compare cluster sizes in each lambda levels
    cluster_sizes_identical = True
    for l in our_lambda_dict.keys():
        if l not in elki_lambda_dict.keys():
            cluster_sizes_identical = False
            print(f"{l} is not an ELKI lambda value")

        else:
            our_cluster_sizes = [len(p['points']) for p in our_lambda_dict[l]].sort()
            elki_cluster_sizes = [len(p['points']) for p in elki_lambda_dict[l]].sort()
            
            if our_cluster_sizes == elki_cluster_sizes:
                print(f"Cluster sizes were identical for lambda={l}")
            if sorted(our_cluster_info[l]['points']) == sorted(elki_cluster_info[l]['points']):
                print(f"Cluster values are identical for lambda={l}")
            else:
                print(f"Cluster sizes or/and values were not identical for lambda={l}")
                cluster_sizes_identical = False


    validated = equal_n_clusters and equal_lambda_amounts and cluster_sizes_identical
    print(f"\nValidation result: The outputs of the algorithms are{' ' if validated else ' not '}identical.") 

    