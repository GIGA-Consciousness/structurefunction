import os
import os.path as op
from surfer import Brain, project_volume_data

subject_id = "fsaverage"
hemi = "lh"
surf = "pial"

brain = Brain(subject_id, hemi, surf)
volume_file = op.abspath("example_fspet/corrected_pet_to_t1/_subject_id_Bend1/r_volume_MGRousset_flirt.nii")
volume_file = op.abspath("ComaSample/data/Bend1/petmr.nii.gz")
pet = project_volume_data(volume_file, hemi,
                            subject_id=subject_id, smooth_fwhm=0.5)
#brain.add_overlay(pet, min=0.1, max=15000)

brain.add_data(pet, min=0, max=60000, thresh=.5,
               colormap="jet", alpha=.6, colorbar=False)