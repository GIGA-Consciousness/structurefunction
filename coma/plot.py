import os
from surfer import Brain, io
subj_id = 'Bend1'
subjects_dir = "/media/BlackBook_/ERIKPETDTIFMRI/ComaSample/subjects"

os.environ['SUBJECTS_DIR'] = subjects_dir

brain = Brain(subj_id, "lh", "pial", subjects_dir=subjects_dir,
              config_opts=dict(background="white"))

"""Project the volume file and return as an array"""
mri_file = "petmr.nii"

surf_data = io.project_volume_data(mri_file, "lh", subject_id=subj_id)
brain.add_data(surf_data, 100, 50000, colormap="jet", alpha=.7)