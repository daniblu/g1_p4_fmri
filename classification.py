from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import permutation_test_score

from nilearn.image import new_img_like, load_img
from nilearn.input_data import NiftiMasker

import numpy as np
import pickle

if __name__ in "__main__":

    # load splits
    with open('splits.pkl', 'rb') as f:
        fmri_img_train, fmri_img_test, conditions_train, conditions_test = pickle.load(f)

    # load searchlight results
    with open('searchlight.pkl', 'rb') as f:
        searchlight, searchlight_scores_ = pickle.load(f)

    # empty list for results
    L = []
    
    # sets of informative voxels
    best_sets = [200.0, 500.0, 1000.0]

    for best in best_sets:

        # find the percentile that makes the cutoff for the X best voxels
        perc = 100*(1-best/searchlight.scores_.size)

        # find the cutoff
        cut = np.percentile(searchlight.scores_, perc)

        # whole brain mask path
        mask_wb_filename = '/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0116/anat/sub-0116_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'

        # load the whole brain mask
        mask_img = load_img(mask_wb_filename)

        # .astype() makes a copy.
        process_mask = mask_img.get_fdata().astype(int)
        process_mask[searchlight.scores_<=cut] = 0
        process_mask_img = new_img_like(mask_img, process_mask)

        masker = NiftiMasker(mask_img=process_mask_img, standardize=False)

        # we use masker to retrieve a 2D array ready for machine learning with scikit-learn
        fmri_masked = masker.fit_transform(fmri_img_test)

        score_cv_test, scores_perm, pvalue = permutation_test_score(
            GaussianNB(), 
            fmri_masked, conditions_test, 
            cv=10, 
            n_permutations=1000, 
            n_jobs=-1, 
            random_state=0, 
            verbose=0, scoring=None)
        
        # write results
        L.append(f"{int(best)}| Classification accuracy: {score_cv_test}, p-value: {pvalue} \n")

    # save results
    with open("out/classification_results.txt", "w") as f:
        f.writelines(L)
