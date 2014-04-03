import glob
from nipype.workflows.dmri.connectivity.group_connectivity import concatcsv

csv_files = glob.glob("*/*.csv")
concatcsv(csv_files)
