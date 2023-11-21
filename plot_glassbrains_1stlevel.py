# script to produce glass brain of 1st level models

import os

import sys
print(sys.executable)

# Import for first-level modeling
from nilearn.glm.first_level import first_level_from_bids

# Additional imports
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import image

# Imports for plotting contrasts
from nilearn.glm import threshold_stats_img
from nilearn import plotting
from nilearn.plotting import plot_stat_map
import scipy.stats as st
import matplotlib

from nilearn.plotting import plot_design_matrix
from nilearn.plotting import plot_contrast_matrix
import seaborn as sns

from scipy.stats import norm
from nilearn.plotting import plot_stat_map
from nilearn.reporting import get_clusters_table
from nilearn.image import mean_img

# load the preprocessed data
#BIDS directory
data_dir='/home/maltelau/cogsci/neuro-23/data/InSpePosNegData/BIDS_2023E/' 
# BIDS derivatives (contains preprocessed data)
derivatives_dir=  '/home/maltelau/cogsci/neuro-23/data/InSpePosNegData/BIDS_2023E/derivatives'  

# Name for experiment in the BIDS directory
task_label = 'boldinnerspeech'
# Label for data that are spatially aligned to the MNI152 template (i.e. spatially normalised)
space_label ='MNI152NLin2009cAsym'
#Run the function that can gather all the needed info from a BIDS folder

def construct_models_from_1st_level(participant = 116, events_sub = ['onset','duration','trial_type']):
    models, models_run_imgs, models_events, models_confounds = \
        first_level_from_bids(
            data_dir, task_label, derivatives_folder=derivatives_dir, n_jobs=6, verbose=0,
            slice_time_ref = None,
            img_filters=[('desc', 'preproc')])
    
    ################
    # Find index for participant 116
    our_participant = [i for i,paths in enumerate(models_run_imgs) if 'sub-0'+str(participant) in paths[0]][0]
    
    ###############
    # Select only the first-level model objects from our participant
    model_part116 = models[our_participant]
    model_run_imgs_part116 = models_run_imgs[our_participant]
    model_events_part116 = models_events[our_participant]
    model_confounds_part116 = models_confounds[our_participant]

    print("Selecting confounds")
    confound_6 = ['trans_x','trans_y','trans_z',
                  'rot_x','rot_y','rot_z',]
    # Subset confounds with selection
    for i in range(len(model_confounds_part116)):
        confounds2=model_confounds_part116[i].copy()
        confounds2=confounds2[confound_6]
        #Removing NAs in the first row.
        confounds2.loc[0,:]=confounds2.loc[1,:]
        model_confounds_part116[i]=confounds2
    print(model_confounds_part116[0].columns)

    # Count number of times the participant correctly pressed the button when stimuli for button press was shown
    for r in range(len(model_events_part116)):
        responses = []
        for i, j in enumerate(model_events_part116[r]['trial_type']):
            if j == "IMG_BI" and model_events_part116[r]['response'][i] == "b":
                responses.append("correct")
            if j == "IMG_BI" and model_events_part116[r]['response'][i] != "b":
                responses.append("incorrect")

        n_corr = responses.count("correct")
        print(f"In run {r} The participant pressed the correct button {n_corr} out of {len(responses)} times")


    # Subset
    for i in range(len(model_events_part116)):
        events1=model_events_part116[i]
        events1=events1[events_sub]
        model_events_part116[i]=events1
        #Rename trials to make contrasting easier
        model_events_part116[i].replace({'IMG_NS': 'N', 'IMG_PS': 'P', 'IMG_NO': 'N', 'IMG_PO': 'P','IMG_BI': 'B'}, inplace = True)
    
    return model_part116, model_run_imgs_part116, model_events_part116, model_confounds_part116


def bonf_corr_plot (newmodel,newcontrast, correction = "bonferroni", comment=''):
    z_map = newmodel.compute_contrast(newcontrast)
    thresholded_map, threshold = threshold_stats_img(
    z_map, alpha=0.05, height_control=correction)
    print('The p<.05 FWER-corrected threshold is %.3g' % threshold)

    filename = 'glassbrain_' + comment + newcontrast + correction + '.jpg'
    filename = "out/" + filename.replace("(", "").replace(")", "").replace("+","").replace("/","")

    plotting.plot_glass_brain(z_map, cmap='PiYG',colorbar=True, threshold=threshold,
                          title=f'Group [{newcontrast}] effect (p<0.05,{correction}-corrected)',
                          plot_abs=False).savefig(filename)
    #plt.show()
    
###################
# construct button press events
models, models_run_imgs, models_events, models_confounds = construct_models_from_1st_level(events_sub = ['onset','duration','trial_type', 'RT'])
    



for i, run in enumerate(models_events):
    for row in range(run.shape[0]):
        if run['trial_type'][row] == 'B':
            onset = run['onset'][row] + run['RT'][row]
            models_events[i] = pd.concat([models_events[i], pd.DataFrame([[onset, 0, 'BP', 0]], columns = run.columns)])

print(models_events[0]['trial_type'].value_counts())
models.fit(models_run_imgs,models_events,models_confounds)

#######
# correlations in design matrix for BP model
# View the design matrix from the first session
design_matrix = models.design_matrices_[0]

# Normalize matrix
normalized_design_matrix = (design_matrix-design_matrix.mean())/design_matrix.std()
design_matrix.rename(columns={"IMG_BI":  "B","IMG_NS": "N", 'IMG_PS': 'P', 'IMG_NO': 'N', 'IMG_PO': 'P'}, inplace = True)
sns.heatmap(design_matrix.corr(), vmin=-1, vmax=1, cmap='RdBu_r').get_figure().savefig("heatmap-buttonpress.png")
#contrasts=['P-N','P+N','B']

bonf_corr_plot(models, 'N-P')
bonf_corr_plot(models, 'N+P')
bonf_corr_plot(models, 'B-(N+P)/2')


bonf_corr_plot(models, 'N-P', 'fdr')
bonf_corr_plot(models, 'N+P', 'fdr')
bonf_corr_plot(models, 'B-(N+P)/2', 'fdr')

###################
# run model withOUT button press events
print("Now run it without BP in model")
models, models_run_imgs, models_events, models_confounds = construct_models_from_1st_level()
models.fit(models_run_imgs,models_events,models_confounds)


bonf_corr_plot(models, 'N-P', comment='covarBP_')
bonf_corr_plot(models, 'N+P', comment= 'covarBP_')
bonf_corr_plot(models, 'B-(N+P)/2', comment= 'covarBP_')

bonf_corr_plot(models, 'N-P', 'fdr', comment= 'covarBP_')
bonf_corr_plot(models, 'N+P', 'fdr', comment= 'covarBP_')
bonf_corr_plot(models, 'B-(N+P)/2', 'fdr', comment= 'covarBP_')

