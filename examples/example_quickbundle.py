import os.path as op
from coma.workflows.dti import bundle_tracks

in_file = op.abspath("/Code/structurefunction/examples/"\
    "precoth/ex_precoth/coma_precoth/_subject_id_Bend1/"\
    "thalamus2precuneus2cortex/converted_intersections_streamline_final.trk")

in_file = op.abspath("ExampleTracks.trk")
out_files = bundle_tracks(in_file)