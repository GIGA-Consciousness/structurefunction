#!/usr/bin/env python
import os, sys
import string

def applyXform(in_file, in_matrix):
    import os.path as op
    from nipype.utils.filemanip import split_filename
    _, name, _ = split_filename(in_file)
    out_basename = name + "_reorient.nii"
    print("Applying transformation")
    import nipype.interfaces.spm.utils as spmu
    applymat = spmu.ApplyTransform()
    applymat.inputs.in_file = op.abspath(in_file)
    applymat.inputs.mat = op.abspath(in_matrix)
    applymat.inputs.out_file = op.abspath(out_basename)
    applymat.run()
    return out_basename


if __name__ == '__main__':
	# Should probably use argparse for this
	in_file, in_matrix = sys.argv[1:]
	applyXform(in_file, in_matrix)
