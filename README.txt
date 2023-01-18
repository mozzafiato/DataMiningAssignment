For reproducing the results, one should have the data from https://osf.io/p2kj7/ (if not already provided), in particular the following files and put them inside the Task2 folder:

ALL_timeseries folder
clinical.csv
FC folder

All of the embeddings are already included, except for the ones for DMGI for the IMDB dataset. In order to save time, one can obtain the neccessary data for DMGI (both FC and IMDB) in this drive: https://drive.google.com/file/d/1-CMDsBNw99gCbFvaf_v8uQKp-iWawTYK/view?usp=share_link. Where the content in the 'data' file should be copied to 'DMGI/data', and the content of the 'embeddings' file should be copied to 'Task2/embeddings' folder (they are not included in the submission because some of them were too large).


Then run the notebooks in the following order:

1. Data preparation 
- constructs the graphs
- applies the two algorithms and saves the embeddings in the specified folder.
(note: We have already included the libraries with the submission, as we have changed/adjusted them in order to avoid potential bugs when running it from scratch)


2. NCut Analysis
- uses the embeddings created in the 'embeddings' folder.
- uses the implementation of NCut from the ncut.py file.
- clusters and analyses the data.

3. IMDB Analysis
(note: for the IMDB-MULTI dataset, one has to download it from https://www.dropbox.com/s/ntutrhk8nr3vveb/imdb.pkl?dl=0 and put it in the 'DMGI/data' folder if not already there)
- applies the two algorithms on the IMDB datasets
- creates additional features and analyses the structure characteristics of the graphs. 

