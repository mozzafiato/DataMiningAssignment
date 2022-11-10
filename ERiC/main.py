import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_neighbourhood_matrix(D, p, k=3):
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


def make_partitions(D):
    point_info = dict()  # useful for later
    partitions = dict()  # store indices of points per partition

    # initialize partitions
    for i in range(D.shape[1]):
        partitions[i+1] = []

    # for every point, compute necessary values and store them
    for i, p in enumerate(D):
        l, e_list, v_list = correlation_dimensionality(D, p)
        point_info[i] = {
            'lambda': l,  # integer
            'E': e_list,  # 1D array
            'V': v_list   # 2D array
        }

        # add index of point to corresponding partition
        partitions[l].append(i)

    return point_info, partitions


###### MAIN ############

D = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
info_point, partitions = make_partitions(D)
print("Info per point:")
print(info_point)
print("Partitions:")
print(partitions)

