import os
import os.path as op
from surfer import Brain, project_volume_data

from coma.datasets import sample
data_path = sample.data_path()

subjects_dir = op.join(data_path, "subjects")
os.environ['SUBJECTS_DIR'] = subjects_dir

subject_id = "Bend1"
hemi = "lh"
surf = "pial"

brain = Brain(subject_id, hemi, surf, subjects_dir=subjects_dir)
volume_file = op.abspath(
    "example_fspet/corrected_pet_to_t1/_subject_id_Bend1/r_volume_MGRousset_flirt.nii")

pet = project_volume_data(volume_file, hemi,
                          subject_id=subject_id, smooth_fwhm=0.5)

brain.add_data(pet, 
               colormap="jet", alpha=.6, colorbar=True)
