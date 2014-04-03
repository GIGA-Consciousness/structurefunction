#!/usr/bin/env python
import os, sys
import string

def JitkaWork(base_name):
    import os.path as op
    from nipype.utils.filemanip import split_filename
    import nipype.interfaces.spm.utils as spmu

    import glob
    list_of_exts = ["cortex","fdgpet", "wmmask"]

    matrix_file = glob.glob("*.mat")[0]

    for filetype in list_of_exts:
        in_file = glob.glob(base_name + filetype + ".*")[0]
        _, name, _ = split_filename(in_file)
        out_basename = name + "_reorient.nii"
        
        print("Applying transformation to %s" % name)
        applymat = spmu.ApplyTransform()
        applymat.inputs.in_file = op.abspath(in_file)
        applymat.inputs.mat = op.abspath(matrix_file)
        applymat.inputs.out_file = op.abspath(out_basename)
        applymat.run()

    return out_basename


if __name__ == '__main__':
	# Should probably use argparse for this
    base_name = sys.argv[1]
    JitkaWork(base_name)
