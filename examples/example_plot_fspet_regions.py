import os
import os.path as op
import scipy.io as sio
import numpy as np
import nibabel as nb
from surfer import Brain

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
data_file = op.abspath("example_fspet/pet_results_npz/_subject_id_Bend1/petmr_pve.npz")

annot_path = op.join(subjects_dir, subject_id, "label")
annot_type = "aparc"
aparc_file = op.join(annot_path, "%s.%s.annot" % (hemi, annot_type))

labels, ctab, aparc_names = nb.freesurfer.read_annot(aparc_file)
print(aparc_names)
print(len(aparc_names))

#method = "PET"
#method = "VOLUMES_(cc)"
method = "MULL-GART_RO"

default_value = 0


data = np.load(data_file)
roi_names = data["region_names"].tolist()
data_to_plot = data[method]
missing_region = False
replace_with = 'WM'

for idx, name in enumerate(roi_names):
    print("%f %s" % (data_to_plot[idx], str(name)))

pos = []
for name in aparc_names:
    # First check if the exact name is in roi_names 
    #if 'unknown' in name:
    #    pos.append(0)

    if name in roi_names:
        pos.append(roi_names.index(name))
    # Otherwise find the name including the hemisphere (e.g. ctx-lh-blah)
    else:
        matching = [s for s in roi_names if name in s and hemi in s]
        if len(matching) > 1:
            pos.append(roi_names.index(matching[0]))
        elif len(matching) == 0:
            print(name + " not found")
            if replace_with in roi_names:
                pos.append(roi_names.index(replace_with))
                print("Replacing with " + replace_with)
            else:
                pos.append(-1)
                missing_region = True
        else:
            pos.append(roi_names.index(matching[0]))

if missing_region:
    data_to_plot = np.hstack((data_to_plot, default_value))

roi_data = data_to_plot[pos]

"""
Make a vector containing the data point at each vertex.
"""
vtx_data = roi_data[labels]

# Check if there are akwardly labelled regions and reset their values
if len(np.where(labels==-1)[0]) > 0:
    vtx_data[np.where(labels==-1)] = default_value

"""
Display these values on the brain. Use a sequential colormap (assuming
these data move from low to high values), and add an alpha channel so the
underlying anatomy is visible.
"""
brain.add_data(vtx_data, min=vtx_data.min(), max=vtx_data.max(), colormap="jet", alpha=.6)

image = brain.save_montage("Example_FDG-PET_FreesurferRegions.png", ['l', 'd', 'm'], orientation='v')

brain.close()

###############################################################################
# View created image
import pylab as pl
fig = pl.figure(figsize=(5, 3), facecolor=bgcolor)
ax = pl.axes(frameon=False)
ax.imshow(image, origin='upper')
pl.draw()
pl.show()