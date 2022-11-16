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
    return sum(eigenvalues[0:r]) / sum(eigenvalues)


def correlation_dimensionality(D, p, k, alpha=0.85):
    # finds lambda λ of the given point p
    # which is the smallest number of eigenvalues of the covariance matrix of N_p
    # explaining a portion of at least α of the total variance

    N_p = get_neighbourhood_matrix(D, p, k)
    eigen_values, eigen_vectors = covariance_decomposition(N_p)
    l = 1
    for i in range(D.shape[1]):
        if total_variance(eigen_values, i+1) >= alpha:
            l = i+1
            break

    return l, eigen_values, eigen_vectors


def make_partitions(D, k):
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
        l, e_list, v_list = correlation_dimensionality(D, p, k)
        # CHANGE since 2 equal points have the same e and v list, p can be used as key
        # CHANGE encode dimensionality in nested index -> saves memory in dbscan alg.
        # caclulate V * E^ * V.T since this value gets used a lot
        VEV = v_list @ E_hats[l - 1] @ v_list.T
        point_info[l][p.data.tobytes()] = {
            # 'lambda': l,  # integer
            'E': e_list,  # 1D array
            'V': v_list,  # 2D array
            'VEV': VEV  # 2D array
        }

        # add index of point to corresponding partition
        partitions[l].append(i)

    return point_info, partitions


def is_approximate_linear_dependant(V_p, VEV_q, delta_affine):
    deltas = np.empty(V_p.shape[1])
    # iterate columns
    for i, v_p in enumerate(V_p.T):
        deltas[i] = v_p @ VEV_q @ v_p.T

    deltas = np.sqrt(deltas)
    # check sqrt(vi.T · Vq · Êq · Vq.T · vi ≤ Δ) for all vi
    return np.all(deltas < delta_affine)


def affine_distance(p, q, VEV_q):
    # compute sqrt((p − q)T · Vq · Êq · Vq.T · (p − q))
    return np.sqrt((p - q).T @ VEV_q @ (p - q))


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

    try:
        # get point info data by hashing point
        x_info = point_info_lx[x.data.tobytes()]
        y_info = point_info_ly[y.data.tobytes()]
    except:
        print("Error")
        print(x)
        print(x.data.tobytes())
        raise ValueError

    # retrieve values to compute similarity
    V_x = x_info['V']
    V_y = y_info['V']
    VEV_x = x_info['VEV']
    VEV_y = y_info['VEV']

    # the condition for 0 distance is:
    # ***
    # SPAN(p) ⊆Δaff SPAN(q) ∧ DISTaff (p, q) ≤ δ ***AND***
    # SPAN(q) ⊆Δaff SPAN(p) ∧ DISTaff (q, p) ≤ δ
    # ***
    # since the maximum value is chosen 

    if (is_approximate_linear_dependant(V_x[:, 0:l_x], VEV_y, delta_affine) and
            affine_distance(x, y, VEV_y) < delta_dist and
            is_approximate_linear_dependant(V_y[:, 0:l_y], VEV_x, delta_affine) and
            affine_distance(y, x, VEV_x) < delta_dist):
        return 0
    else:
        return 1


def cluster_partitions(
        D, partitions, point_info,
        delta_affine, delta_dist, min_samples
):
    # initialize output structures
    models = [None for _ in partitions]  # maybe useful later?
    clusters = {i: [] for i in partitions}

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
                eps=np.nextafter(0, 1),
                min_samples=min_samples,
                metric='precomputed',
            ).fit(X)

            models[l] = model

        
            # get indices from model
            label = 0
            # iterate labels
            while True:
                # get indices of partition where
                # points are clustered the current label
                cluster = np.transpose((model.labels_ == label).nonzero())
                if cluster.size != 0:
                    clusters[l].append(cluster)
                    label += 1
                else:
                    # stop if model has no more labels
                    break
        

    return models, clusters
