This folder contains our implementation of the ERiC algorithm.

## Requirements
In order to run the EDA, the following requirements must be fulfilled:
- The folders "auxiliarymethods", "kernels" and "datasets" and their content must be in the same directory as the program files. These can be found in the https://github.com/chrsmrrs/tudataset.git repository inside the "tudataset/tud_benchmark" folders. 
- The Java binary for ELKI ERiC (see https://elki-project.github.io/releases/release0.7.5/elki-bundle-0.7.5.jar) must be in the same directory as the program files.
- The sample datasets from https://elki-project.github.io/datasets/ must be in the folder "sample_datasets" in the same directory as the program files. 
- TODO: libraries?

## Files
- **EDA.ipynb** contains the exploratory data analysis with methods such as dimensionality reduction, clustering etc.
- **EDA_ERiC.ipynb** visualizes and analyzes the clusterings obtained with ERiC.
- **ERiC_validation.ipynb** contains methods for the validation of ERiC, comparing it with the ELKI implementation.
