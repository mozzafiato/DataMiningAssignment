import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN


def get_neighbourhood_matrix(D, p, k=3):
    # compute matrix N_p of shape k x d
    nbrs = NearestNeighbors(
        n_neighbors=k, algorithm='brute', metric="euclidean").fit(D)
    _, indices = nbrs.kneighbors([p])
    return D[indices[0]]


def get_weak_eigenvectors_matrix(meta):
    pass

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
    # calculates the total explained varience of using
    # the first r eigenvalues
    return sum(eigenvalues[0:r])/sum(eigenvalues)


def correlation_dimensionality(D, p, alpha=0.85):
    # finds lambda λ of the given point p
    # which is the smallest number of eigenvalues of the covariance matrix of N_p
    # explaining a portion of at least α of the total variance

    N_p = get_neighbourhood_matrix(D, p)
    eigen_values, eigen_vectors = covariance_decomposition(N_p)
    l = 1
    for i in range(D.shape[1]):
        if total_variance(eigen_values, i) >= alpha:
            l = i
            break

    return l, eigen_values, eigen_vectors


def make_partitions(D, k=1000):
    point_info = dict()  # useful for later
    partitions = dict()  # store indices of points per partition

    # initialize partitions
    for i in range(D.shape[1]):
        point_info[i+1] = dict()
        partitions[i+1] = []

    # for every point, compute necessary values and store them
    for i, p in enumerate(D):
        l, e_list, v_list = correlation_dimensionality(D, p, k)
        # CHANGE since 2 equal points have the same e and v list, p can be used as key
        # CHANGE encode dimensionality in nested index -> saves memory in dbscan alg.
        point_info[l][p] = {
            # 'lambda': l,  # integer
            'E': e_list,  # 1D array
            'V': v_list   # 2D array
        }

        # add index of point to corresponding partition
        partitions[l].append(i)

    return point_info, partitions

def is_approximate_linear_dependant(V_p, VEV_q, delta_affine):

    deltas = np.sqrt(V_p @ VEV_q @ V_p.T)

    return np.all(deltas < delta_affine)

def affine_distance(p, q, VEV_q):
    return np.sqrt((p-q).T @ VEV_q @ (p-q))

def symmetric_correlation_distance(
    x, y, delta_affine, delta_dist,
    lx, point_info_lx,
    ly=None, point_info_ly=None # distances of different dimensionalities will be useful later
):

    d = x.size

    if point_info_ly is None:
        point_info_ly = point_info_lx

    E_hat_x = np.eye(d)
    E_hat_x[0:lx, 0:lx] = 0
    if ly is not None:
        E_hat_y = np.eye(d)
        E_hat_y[0:ly, 0:ly] = 0
    else:
        E_hat_y = E_hat_x

    V_x = point_info_lx[x]['V']
    V_y = point_info_ly[y]['V']

    VEV_x = V_x @ E_hat_x @ V_x.T
    VEV_y = V_y @ E_hat_y @ V_y.T

    if (is_approximate_linear_dependant(V_x, VEV_y, delta_affine) and
            is_approximate_linear_dependant(V_y, VEV_x, delta_affine) and
            affine_distance(x, y, VEV_y) < delta_dist and
            affine_distance(y, x, VEV_x) < delta_dist):
        return 0
    else:
        return 1

def cluster_partitions(
    partitions, point_info,
    delta_affine, delta_dist, min_samples
):
    models = []
    for l, p in partitions.items():
        metric_params = (
            delta_affine, delta_dist,
            l, point_info[l],
        )
        model = DBSCAN(
            0, min_samples, symmetric_correlation_distance, metric_params
        ).fit(p)

        models.append(model)