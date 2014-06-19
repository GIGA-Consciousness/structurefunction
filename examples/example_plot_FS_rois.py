import os
import os.path as op
from surfer import Brain, io

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

annot_path = op.join(subjects_dir, subject_id, "label", "%s.aparc.annot" % hemi)
assert(op.exists(annot_path))
brain.add_annotation(annot_path, borders=False)

image = brain.save_montage("Example_FreesurferRegions.png", ['l', 'd', 'm'], orientation='v')
brain.close()
