# Group 1 portfolio 4
## Advanced cognitive neuroscience E23

This repository contains the code to reproduce the analysis of study group 1 portfolio 4.

## Overview
- __plot_glassbrains_1stlevel.py__: Creates plots of first level model contrasts for two different correction methods (false detection rate and Bonferroni). Saves the plots in ```out```.

- __searchlight.py__: Loads in the fmri data _of one subject_, creates an LSA design matrix, fits a GLM for each voxel, extracts OLS beta values, saves the train-test split of the betas as ```splits.pkl``` performs, searchlight analysis and saves the results as ```searchlight.pkl```.

- __plot_searchlight.py__: Saves plots of searchlight scores and most informative voxels in ```out```.

- __label_clusters.py__: Makes use of the module ```atlasreader``` to label the clusters of most informative voxels. Plots indicating the position of each detected cluster as well as tables are saved in ```out/atlasreader```.

- __classification.py__: Loads ```splits.pkl``` and ```searchlight.pkl``` to sumbit selected data to a classification algorithm and save the classification results in ```out/classification_results.txt```.

## Setup
Run the command below in a terminal to setup and environment and dowload the modules listed in ```requirements.txt```.
```shell
bash setup_env.sh
```

Remember to activate the environment before running the scripts above.
```shell
source env/bin/activate
```

