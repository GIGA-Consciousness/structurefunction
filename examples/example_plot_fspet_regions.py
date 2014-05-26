import os
import os.path as op
import scipy.io as sio
import nibabel as nb
from surfer import Brain

from coma.datasets import sample
data_path = sample.data_path()

subjects_dir = op.join(data_path, "subjects")
os.environ['SUBJECTS_DIR'] = subjects_dir

subject_id = "Bend1"
hemi = "lh"
surf = "pial"

#brain = Brain(subject_id, hemi, surf, subjects_dir=subjects_dir)
data_file = op.abspath("example_fspet/pet_stats/_subject_id_Bend1/stats.mat")

annot_path = op.join(subjects_dir, subject_id, "label")
annot_type = "aparc"
aparc_file = op.join(annot_path, "%s.%s.annot" % (hemi, annot_type))

labels, ctab, aparc_names = nb.freesurfer.read_annot(aparc_file)
print(aparc_names)
print(len(aparc_names))


data = sio.loadmat(data_file)

pos = []
roi_names = data['roi_names'].tolist()
for name in aparc_names:
    # First check if the exact name is in roi_names 
    #if 'unknown' in name:
    #    import ipdb
    #    ipdb.set_trace()
    #    pos.append(0)

    if name in roi_names:
        pos.append(roi_names.index(name))
    # Otherwise find the name including the hemisphere (e.g. ctx-lh-blah)
    else:
        matching = [s for s in roi_names if name in s]
        if len(matching) > 1:
            matching = [s for s in roi_names if name in s and hemi in s]
        if len(matching) == 0:
            pass
        else:
            pos.append(roi_names.index(matching[0]))

for idx, name in enumerate(roi_names):
    print("%f %s %d" % (data["func_mean"][idx][0], str(name), data["rois"][0][idx]))

roi_data = data['func_mean'][pos]

"""
Make a vector containing the data point at each vertex.
"""
vtx_data = roi_data[labels]

"""
Display these values on the brain. Use a sequential colormap (assuming
these data move from low to high values), and add an alpha channel so the
underlying anatomy is visible.
"""
brain.add_data(vtx_data, colormap="GnBu", alpha=.8)
