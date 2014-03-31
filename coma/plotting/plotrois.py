#Run with  ipython --gui=wx


from os import environ
from os.path import join
import numpy as np
from surfer import Brain, io

import os
import os.path as op
import numpy.random as random


subject_id = 'fsaverage'
hemi = "rh"
surf = "pial"
bkrnd = "white"

brain = Brain(subject_id, hemi, surf, config_opts=dict(background=bkrnd))
brain.add_annotation(op.abspath("%s.Lausanne1015_fsavg.annot" % hemi), borders=False)
