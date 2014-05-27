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
bgcolor = 'w'

brain = Brain(subject_id, hemi, surf, config_opts={'background': bgcolor},
    subjects_dir=subjects_dir)
volume_file = op.abspath("example_fspet/corrected_pet_to_t1/_subject_id_Bend1/r_volume_MGRousset_flirt.nii")

pet = project_volume_data(volume_file, hemi,
                          subject_id=subject_id)

brain.add_data(pet, min=250, max=12000,
               colormap="jet", alpha=.6, colorbar=True)

image = brain.save_montage("Example_FDG-PET.png", ['l', 'd', 'm'], orientation='v')

brain.close()

###############################################################################
# View created image
import pylab as pl
fig = pl.figure(figsize=(5, 3), facecolor=bgcolor)
ax = pl.axes(frameon=False)
ax.imshow(image, origin='upper')
pl.draw()
pl.show()