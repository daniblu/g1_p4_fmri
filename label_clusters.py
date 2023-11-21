from atlasreader import create_output
from nilearn.image import new_img_like
import numpy as np
import pandas as pd
import pickle

if __name__ in "__main__":
    
    # load searchlight results
    with open('searchlight.pkl', 'rb') as f:
        searchlight, searchlight_scores_ = pickle.load(f)

    # load anatomical image
    anat_filename = '/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0116/anat/sub-0116_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'

    # find the percentile that makes the cutoff for the 200 best voxels
    perc = 100*(1-200.0/searchlight.scores_.size)

    # find the cutoff
    cut = np.percentile(searchlight.scores_, perc)

    # create an image of the searchlight scores
    searchlight_img = new_img_like(anat_filename, searchlight.scores_)

    # create output
    create_output(searchlight_img, voxel_thresh=cut, cluster_extent=5, direction='both', 
    outdir='out/atlasreader')
