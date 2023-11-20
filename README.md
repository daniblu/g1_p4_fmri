# Group 1 portfolio 4
## Advanced cognitive neuroscience E23

This repository contains the code to reproduce the analysis of study group 1 portfolio 4.

## Overview
This script 
```shell
python3 SANITY CHECK
```

This script loads in the fmri data _of one subject_, creates an LSA design matrix, fits a GLM for each voxel, extracts OLS beta values, performs searchlight analysis and saves the results as ```searchlight.pkl```.
```shell
python3 searchlight.py
```
