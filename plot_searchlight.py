from nilearn.plotting import plot_glass_brain
from nilearn.image import new_img_like
import numpy as np
import pickle


if __name__ in "__main__":

    # load searchlight results
    with open('searchlight.pkl', 'rb') as f:
        searchlight, searchlight_scores_ = pickle.load(f)
    
    with open('searchlight_n_b.pkl', 'rb') as f:
        searchlight2, searchlight2_scores_ = pickle.load(f)
    
    # load anatomical image
    anat_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0116/anat/sub-0116_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'

    # PLOT SEARCHLIGHT ACCURACIES

    # negative versus positive
    searchlight_img = new_img_like(anat_filename, searchlight_scores_)

    fig = plot_glass_brain(searchlight_img, cmap='prism', colorbar=True, threshold=0.76, title='negative vs positive (Acc>0.76)')
    fig.savefig("out/pos_neg_searchlight.png", dpi=300)

    # negative versus button
    searchlight_img = new_img_like(anat_filename, searchlight2_scores_)

    fig = plot_glass_brain(searchlight_img, cmap='prism', colorbar=True, threshold=0.76, title='negative vs button (Acc>0.76)')
    fig.savefig("out/neg_but_searchlight.png", dpi=300)

    # PLOT TOP VOXELS
    best_sets = [200.0, 500.0, 1000.0]
    titles = ["A) 200 most informative voxels", "B) 500 most informative voxels", "C) 1000 most informative voxels"]

    for i, best in enumerate(best_sets):
        # find the percentile that makes the cutoff for the X best voxels
        perc=100*(1-best/searchlight.scores_.size)

        # find the cutoff
        cut=np.percentile(searchlight.scores_,perc)

        # create an image of the searchlight scores
        searchlight_img = new_img_like(anat_filename, searchlight.scores_)

        # plot
        fig = plot_glass_brain(searchlight_img, threshold=cut, title=titles[i])
        fig.savefig(f"out/pos_neg_searchlight_{int(best)}.png", dpi=300)