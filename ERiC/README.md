### Team 3
**Authors:** Kraljevska Melanija, Stolz Michaela, Vincent Jonathan, Zimmball Markus
This folder contains our implementation of the ERiC algorithm and its evaluation on the IMDB-BINARY dataset.

## Requirements
In order to run the programs, the following requirements must be fulfilled:
- The folders "auxiliarymethods", "kernels" and "datasets" and their content must be in the same directory as the program files. These can be found in the https://github.com/chrsmrrs/tudataset.git repository inside the "tudataset/tud_benchmark" folders. 
- The Java binary for ELKI ERiC (see https://elki-project.github.io/releases/release0.7.5/elki-bundle-0.7.5.jar) must be in the same directory as the program files.
- The sample datasets from https://elki-project.github.io/datasets/ must be in the folder "sample_datasets" in the same directory as the program files. 
- In order to import auxiliarymethods, pytorch-geometric must be installed:
```
Install pytorch geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

 Here is the gpu cuda installation, for the cpu version replace cu102 with cpu
 ! pip install torch-scatter==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
 ! pip install torch-sparse==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
 ! pip install torch-cluster==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
 ! pip install torch-spline-conv==latest+cu102 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
 ! pip install torch-geometric
```

## Reports of results: 
- **EDA.ipynb** contains the exploratory data analysis with methods such as dimensionality reduction, clustering etc.
- **EDA.html** containts the results of EDA.ipynb.
- **EDA_ERiC.ipynb** visualizes and analyzes the clusterings obtained with ERiC.
- **EDA_ERiC.html** containts the results of EDA_ERiC.ipynb.

## Computation of results: 
- **kernel_computation.ipynb** computes the kernels and stores the results in pickle files.
- **Run_ERiC.ipynb** runs ERiC on a given dataset and kernel and stores the result in pickle files.
- **ERiC_Validation.ipynb** contains code for the validation of ERiC, comparing it with the ELKI implementation.
- **plot_trees.ipynb** plots the cluster hierarchies for different combinations of hyperparameters.

## Helper files:
- **lib.py** contains the implementation of ERiC.
- **elki_eric.py** runs the ELKI implementation of ERiC and saves in a text file.
- **elki_parser.py** parses the results of ELKI ERiC and contains a function for visualization.
- **validation.py** compares two outputs of ERiC for validation.
