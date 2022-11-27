# Script for running an elki algorithm and save its results, here with an example for the COPAC algorithm
import os
import re
from subprocess import Popen, PIPE
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler as Scaler
import re

DATA_FILE_NAME = "data.tsv"
# Install Java and download the elki bundle https://elki-project.github.io/releases/release0.7.5/elki-bundle-0.7.5.jar
ELKI_JAR = "elki-bundle-0.7.5.jar"


def elki_eric(X, k=100, dbscan_minpts=4, alpha=0.85, delta_dist=0.5, delta_affine=1.5, output_file_name=None):
    """Perform ERIC clustering implemented by ELKI package.
       The function calls jar package, which must be accessible through the
       path stated in ELKI_JAR constant.

    """

    np.savetxt(DATA_FILE_NAME, X, delimiter=",", fmt="%.6f")
    print("Run elki")
    # run elki with java
    # You can find the read of the names of the parameters for the ERiC algorithm from the elki GUI
    process = Popen(["java", "-cp", ELKI_JAR, "de.lmu.ifi.dbs.elki.application.KDDCLIApplication",
                     "-algorithm", "clustering.correlation.ERiC",
                     "-dbc.in", "data.tsv",
                     "-parser.colsep", ",",
                     "-eric.k", str(k),
                     "-ericdf.delta", str(delta_dist),
                     "-ericdf.tau", str(delta_affine),
                     "-dbscan.minpts", str(dbscan_minpts),
                     "-pca.filter.alpha", str(alpha)],
                    stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code != 0:
        raise IOError("Elki implementation failed to execute: \n {}".format(output.decode("utf-8")))

    # remove data file
    os.remove(DATA_FILE_NAME)

    # parse output
    elki_output = output.decode("utf-8")

    if output_file_name is None:
        output_file_name = 'elki_eric_output.txt'

    print("Saving ELKI results in", output_file_name)
    with open(output_file_name, 'w') as f:
        f.write(re.sub(r'\n\s*\n', '\n', elki_output, re.MULTILINE))
    print("Writing completed.")


if __name__ == "__main__":

    df = pd.read_csv("sample_dataset/wages.csv")
    df = df[["AGE", "EDUCATION", "EXPERIENCE", "WAGE"]]
    df.columns = ["A", "YE", "YW", "W"]
    print(df.head())
    D = df.to_numpy(dtype=np.float64)
    D = Scaler().fit_transform(D)

    elki_eric(D)

    """
    x, y = make_blobs(n_samples=200,
                      n_features=2,
                      centers=2,
                      cluster_std=0.1,
                      )
    print(x.shape)
    print(y.shape)
    elki_eric(x)

    nmi = normalized_mutual_info_score(pred, y)
    print(f"NMI: {nmi:.4f}")
    pred_and_y = np.concatenate([pred[:, None], y[:, None]], axis=1)
    #np.savetxt("results.csv", pred_and_y, delimiter=",", fmt="%.1f")
    """

