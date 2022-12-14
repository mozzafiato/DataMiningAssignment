import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform


def get_neighbourhood_matrix(D, p, k=200):
    # compute matrix N_p of shape k x d
    nbrs = NearestNeighbors(
        n_neighbors=k, algorithm='brute', metric="euclidean").fit(D)
    _, indices = nbrs.kneighbors([p])
    return D[indices[0]]


def covariance(X):
    # covariance matrix of shape dxd
    return np.cov(X, rowvar=False)


def covariance_decomposition(X):
    # compute covariance
    X_cov = covariance(X)

    # compute eigen values and vectors
    eigen_values, eigen_vectors = np.linalg.eig(X_cov)

    # sort by biggest eigenvalue
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    return eigen_values, eigen_vectors


def total_variance(eigenvalues, r):
    # calculates the total explained variance of using
    # the first r eigenvalues
    return sum(eigenvalues[0:r+1]) / sum(eigenvalues)


def correlation_dimensionality(D, p, k, alpha=0.85, i=None):
    # finds lambda λ of the given point p
    # which is the smallest number of eigenvalues of the covariance matrix of N_p
    # explaining a portion of at least α of the total variance

    N_p = get_neighbourhood_matrix(D, p, k)
    eigen_values, eigen_vectors = covariance_decomposition(N_p)

    l = 0
    for i in range(D.shape[1]):
        if total_variance(eigen_values, i) >= alpha:
            l = i
            break

    return l+1, eigen_values, eigen_vectors


def make_partitions(D, k, alpha=.85):
    point_info = dict()  # useful for later
    partitions = dict()  # store indices of points per partition
    E_hats = []  # diagonal weak eigenvector filters

    # initialize partitions, meta data and eigenvector filter
    for i in range(D.shape[1]):
        point_info[i + 1] = dict()
        partitions[i + 1] = []
        # weak eigenvector filter used to compute 
        # Vq · Êq · VqT
        E_hat = np.eye(D.shape[1])
        E_hat[0:i + 1, 0:i + 1] = 0
        E_hats.append(E_hat)

    # for every point, compute necessary values and store them
    for i, p in enumerate(D):
        l, e_list, v_list = correlation_dimensionality(D, p, k, alpha, i)
        # CHANGE since 2 equal points have the same e and v list, p can be used as key
        # CHANGE encode dimensionality in nested index -> saves memory in dbscan alg.
        # caclulate V * E^ * V.T since this value gets used a lot
        VEV = v_list @ E_hats[l - 1] @ v_list.T
        point_info[l][p.data.tobytes()] = {
            'V': v_list,  # 2D array
            'VEV': VEV  # 2D array
        }

        # add index of point to corresponding partition
        partitions[l].append(i)

    return point_info, partitions


def is_approximate_linear_dependant(V_p, VEV_q, delta_affine):
    # iterate columns
    for i, v_p in enumerate(V_p.T):
        p = v_p @ VEV_q @ v_p.T
        deltas_i = np.sqrt(np.abs(p))
        if deltas_i > delta_affine: return False

    return True


def affine_distance(p, q, VEV_q):
    # compute sqrt((p − q)T · Vq · Êq · Vq.T · (p − q))
    pr = max(0, (p - q).T @ VEV_q @ (p - q))
    return np.sqrt(pr)


def corr_distance(p, q, V_p, VEV_q, delta_dist, delta_affine):
    """"
    checks SPAN(p) ⊆Δaff SPAN(q) ∧ DISTaff (p, q) ≤ δ
    """
    if is_approximate_linear_dependant(V_p, VEV_q, delta_affine) \
            and affine_distance(p, q, VEV_q) < delta_dist:
        return 0
    else:
        return 1


def symmetric_correlation_distance(
        x, y, delta_affine, delta_dist,
        l_x, point_info_lx,
        l_y=None, point_info_ly=None  # distances of different dimensionalities will be useful later
):
    # point info dictionaries for dimensionalities lx and ly
    # check if points have same lambda dimensionality
    if point_info_ly is None:
        l_y = l_x
        point_info_ly = point_info_lx

    # get point info data by hashing point
    x_info = point_info_lx[x.data.tobytes()]
    y_info = point_info_ly[y.data.tobytes()]

    # retrieve values to compute similarity
    V_x = x_info['V']
    V_y = y_info['V']
    VEV_x = x_info['VEV']
    VEV_y = y_info['VEV']

    return max(corr_distance(x, y, V_x[:, 0:l_x], VEV_y, delta_dist, delta_affine),
               corr_distance(y, x, V_y[:, 0:l_y], VEV_x, delta_dist, delta_affine))


def cluster_partitions(
        D, partitions, point_info,
        delta_affine, delta_dist, min_samples
):
    # initialize output structures
    models = [None for _ in partitions]  # maybe useful later?
    clusters = {i: [] for i in partitions}
    # initialize noise
    clusters[D.shape[1]].append([])

    for l, p in partitions.items():
        # check if partition contains indices
        if p:
            # metric params contains the point info dictionary
            # of dimensionality lamda=l
            metric_params = {
                'delta_affine': delta_affine,
                'delta_dist': delta_dist,
                'point_info_lx': point_info[l],
                'l_x': l
            }

            # pairwise distances
            X = pdist(
                D[p],
                lambda x, y: symmetric_correlation_distance(x, y, **metric_params),
            )
            X = squareform(X)

            # perform DBSCAN
            # eps is the closest value to zero,
            # since we have a binary similarity function
            model = DBSCAN(
                eps=0.0000000001,
                min_samples=min_samples,
                metric='precomputed',
            ).fit(X)

            # models[l] = model

            # get indices from model
            label = 0
            # iterate labels
            for label in np.unique(model.labels_):
                # get points of the current cluster
                if label == -1 or l == D.shape[1]:
                    # put points marked as noise by ERiC or DBSCAN in the last (noise) partition
                    clusters[D.shape[1]][0].extend(np.asarray(p)[model.labels_ == label])
                else:
                    clusters[l].append(np.asarray(p)[model.labels_ == label])

    return models, clusters


def compute_cluster_list(clusters, D):
    # nested dictionary: 1.key: partition, 2.key: cluster
    cluster_info = {}
    c_i = 1

    # create cluster info dictionary for every cluster in all partitions
    for p in list(clusters.keys()):

        if len(clusters[p]) > 0:
            for c in range(1, len(clusters[p]) + 1):
                cluster_info[c_i] = {}
                cluster_info[c_i]['lambda'] = p
                cluster_info[c_i]['points'] = clusters[p][c - 1]
                cluster_info[c_i]['index'] = c - 1

                # compute centroid of cluster c in partition p
                N_cluster = np.squeeze(D[clusters[p][c - 1]])
                cluster_info[c_i]['centroid'] = np.mean(N_cluster, axis=0)

                # compute matrices based on strong and weak eigenvalues
                e_list, v_list = covariance_decomposition(N_cluster)
                E_hat = np.eye(N_cluster.shape[1])
                E_hat[0:p, 0:p] = 0
                cluster_info[c_i]['V'] = np.real(v_list)
                cluster_info[c_i]['VEV'] = v_list @ E_hat @ v_list.T

                # initialize parents
                cluster_info[c_i]['parents'] = []

                c_i += 1

    return cluster_info


def is_parent(j, i, cluster_list):
    # checks if c_j is a (grand)parent of c_i
    for c in cluster_list.keys():
        if c in cluster_list[i]["parents"]:
            if j in cluster_list[c]["parents"]:
                return True
    return False


def build_hierarchy(cluster_list, delta_affine, delta_dist, d):

    l_max = d

    for i in cluster_list.keys():
        c_i = cluster_list[i]
        l_ci = c_i['lambda']
        for j in cluster_list.keys():
            c_j = cluster_list[j]
            l_cj = c_j['lambda']

            if l_ci < l_cj:
                if l_cj == l_max and len(c_i['parents']) == 0:
                    c_i['parents'].append(j)
                else:
                    cent_i = c_i['centroid']
                    cent_j = c_j['centroid']
                    V_i = c_i['V']
                    VEV_j = c_j['VEV']

                    if corr_distance(cent_i, cent_j, V_i, VEV_j, delta_dist, delta_affine) == 0 \
                            and (len(c_i['parents']) == 0 or not is_parent(j, i, cluster_list)):
                        c_i['parents'].append(j)

    return cluster_list
