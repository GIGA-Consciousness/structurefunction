print __doc__

import os
import os.path as op
import numpy as np
import nibabel as nb
from surfer import Brain

subject_id = "fsaverage"
hemi = "rh"
surface = "pial"

"""
Bring up the visualization.
"""
brain = Brain(subject_id, hemi, surface,
              config_opts=dict(background="white"))

"""
Read in the annot file
"""
aparc_file = op.abspath("%s.Lausanne1015_fsavg.annot" % hemi)
labels, ctab, names = nb.freesurfer.read_annot(aparc_file)
print(names)
print(len(names))

"""
Make a random vector of scalar data corresponding to a value for each region in
the parcellation.

"""
rs = np.random.randint(0,2,size=len(names))
roi_data = rs

"""
Make a vector containing the data point at each vertex.
"""
vtx_data = roi_data[labels]

"""
Display these values on the brain. Use a sequential colormap (assuming
these data move from low to high values), and add an alpha channel so the
underlying anatomy is visible.
"""
brain.add_data(vtx_data, 0, 2, colormap="GnBu", alpha=.8)
