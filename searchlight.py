import os

import nilearn
from nilearn.glm.first_level import first_level_from_bids
from nilearn.glm.first_level import FirstLevelModel
from nilearn import masking
from nilearn.image import load_img, index_img, concat_imgs

from sklearn.model_selection import train_test_split

import nibabel as nib

import numpy as np

from nilearn import decoding
from nilearn.decoding import SearchLight

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

import pickle

if __name__ == '__main__':

    # BIDS directory
    bids_dir = '/work/816119/InSpePosNegData/BIDS_2023E/' 

    # BIDS derivatives (contains preprocessed data)
    derivatives_dir = os.path.join(bids_dir, 'derivatives')  

    # name for experiment in the BIDS directory
    task_label = 'boldinnerspeech'

    # label for data that are spatially aligned to the MNI152 template (i.e. spatially normalized)
    space_label ='MNI152NLin2009cAsym'

    # run the function that can gather all the needed info from a BIDS folder
    models, models_run_imgs, models_events, models_confounds = \
        first_level_from_bids(bids_dir, task_label, 
                              derivatives_folder=derivatives_dir, 
                              n_jobs=6, verbose=0,
                              img_filters=[('desc', 'preproc')])
    
    # identify the index of subject 116
    our_participant = [i for i, paths in enumerate(models_run_imgs) if 'sub-0116' in paths[0]][0]

    # modify events
    events_sub= ['onset','duration','trial_type']

    for ii in range(len(models_events)):
        events1=models_events[ii][:]
        for i in range(len(events1)):
            events2=events1[i]
            events2=events2[events_sub]
            events1[i]=events2
            
            # rename trials to make contrasting easier (ignore self-other dimension)
            events1[i].replace({'IMG_NS': 'N', 'IMG_PS': 'P', 'IMG_NO': 'N', 'IMG_PO': 'P','IMG_BI': 'B'}, inplace = True);

        models_events[ii][:]=events1
    
    # modify confounds
    selection = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']

    for ii in range(len(models_confounds)):
        confounds1=models_confounds[ii][:].copy()
        for i in range(len(confounds1)):
            confounds2=confounds1[i].copy()
            confounds2=confounds2[selection]
            #Removing NAs in the first row.
            confounds2.loc[0,:]=confounds2.loc[1,:]
            confounds1[i]=confounds2
        models_confounds[ii][:]=confounds1
    
    # get data and model info for our participant
    model1=models[our_participant]
    imgs1=models_run_imgs[our_participant]
    events1=models_events[our_participant]
    confounds1=models_confounds[our_participant]

    # create paths to masks of each run and merge
    subject = "0116"
    runs = [1,2,3,4,5,6]
    
    fprep_func_dir  = os.path.join(bids_dir, "derivatives", f"sub-{subject}", "func")
    mask_paths = [os.path.join(fprep_func_dir, f"sub-{subject}_task-boldinnerspeech_run-{run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz") for run in runs]
    masks = [nib.load(path) for path in mask_paths]
    mask_img = masking.intersect_masks(masks, threshold=0.8)

    # rename trial_type in events
    events1_lsa = events1
    for run in range(len(events1_lsa)):
        events_temp = events1_lsa[run]
        events_temp['trial_type'] = [f"trial_{i+1}_{t}" for i, t in enumerate(events_temp['trial_type'])]
        events1_lsa[run] = events_temp
    
    # initiate model
    flm = FirstLevelModel(
        t_r=1,
        slice_time_ref=0.5,
        mask_img=mask_img,
        hrf_model='glover',
        drift_model='cosine',
        high_pass=0.01,
        smoothing_fwhm=None,
        minimize_memory=True,
        noise_model='ols'
    )

    # fit model (involves creating design matrices). Since trial_types have different orders in each run we fit a model for each run seperately.
    print("[INFO]: Fitting models for each block/run")
    keys = ['run', 'model']
    flms = {key: [] for key in keys}

    for run in runs:
        flms['run'].append(run)
        flm.fit(imgs1[run-1], events = events1_lsa[run-1], confounds = confounds1[run-1]);
        flms['model'].append(flm)
    
    # create matrix containing contrast vectors (the rows)
    N_all = flms['model'][0].design_matrices_[0].shape[1]
    contrasts = np.eye(N_all)

    # keep only contrast vectors/rows corresponding to the trial variables
    N_trials = len(events1[0]['trial_type'])
    contrasts = contrasts[:N_trials, :]

    # extract betas for the whole volume
    print("[INFO]: Extracting betas")
    pattern = []
    conditions_label = []

    for run in runs:
        
        for i in range(contrasts.shape[0]):

            # extract all betas for one contrast for all voxels
            betas_img = flms['model'][run-1].compute_contrast(contrasts[i,:], output_type='effect_size')
            pattern.append(betas_img)
            
            # extract condition label
            conditions_label.append(flms['model'][run-1].design_matrices_[0].columns[i][-1])
        

    pattern = concat_imgs(pattern)

    # keep only positive and negative conditions
    idx_neg=[int(i) for i in range(len(conditions_label)) if 'N' in conditions_label[i]]
    idx_pos=[int(i) for i in range(len(conditions_label)) if 'P' in conditions_label[i]]
    idx_but=[int(i) for i in range(len(conditions_label)) if 'B' in conditions_label[i]]

    idx=np.concatenate((idx_neg, idx_pos))

    conditions=np.array(conditions_label)[idx]

    pattern_p_n = index_img(pattern, idx)

    # make an index for spliting fMRI data
    idx2 = np.arange(conditions.shape[0])

    # create training and testing indeces and conditions
    idx_train, idx_test, conditions_train, conditions_test = train_test_split(idx2, conditions, test_size=0.2)

    # split pattern according to train and text indices
    fmri_img_train = index_img(pattern_p_n, idx_train)
    fmri_img_test = index_img(pattern_p_n, idx_test)
    
    # save splits
    with open('splits.pkl', 'wb') as f:
        pickle.dump([fmri_img_train, fmri_img_test, conditions_train, conditions_test], f)

    # load mask image
    mask_wb_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0116/anat/sub-0116_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
    mask_img = load_img(mask_wb_filename)

    # searchlight
    print("[INFO]: Searchlight analysis")

    searchlight = SearchLight(
    mask_img,
    estimator=GaussianNB(),
    radius=5,
    n_jobs=-1,
    verbose=10, 
    cv=10)
    
    searchlight.fit(fmri_img_train, conditions_train)

    # save searchlight output
    with open('searchlight.pkl', 'wb') as f:
        pickle.dump([searchlight, searchlight.scores_], f)

