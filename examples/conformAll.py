#!/usr/bin/env python
import os, sys
import string

def conformAll(base_name):
    import os.path as op
    import nipype.interfaces.freesurfer as fs
    from nipype.utils.filemanip import split_filename
    import glob
    list_of_exts = ["T1"    , "T1brain", "PreCoTh_rois",
    "fdgpet_reorient", "wmmask_reorient" , "cortex_reorient",
    "fdgpet", "wmmask" , "cortex"]
    out_files = []

    print("Base name: '%s'" % base_name)

    try:
        os.mkdir('Original')
    except OSError:
        print("Directory exists")

    for filetype in list_of_exts:
        print("Searching for '%s'" % filetype)
        search_str = op.abspath(base_name + "*" + filetype + ".*")
        in_file = glob.glob(search_str)
        if in_file and len(in_file) == 1:
            in_file = in_file[0]
            print("Found %s" % in_file)
            _, name, ext = split_filename(in_file)
            out_file = op.abspath("conform_" + name + ".nii.gz")
            if filetype == "fdgpet" or filetype == "wmmask" or filetype == "cortex":
                cmd = 'mv %s %s' % (in_file, op.join(op.abspath("Original"), name + ext))
                os.system(cmd)
                print(cmd)
            else:
                conv = fs.MRIConvert()
                conv.inputs.conform = True
                conv.inputs.no_change = True
                if filetype == "PreCoTh_rois" or filetype == "wmmask_reorient" or filetype == "cortex_reorient":
                    conv.inputs.resample_type = 'nearest'
                conv.inputs.in_file = in_file
                conv.inputs.out_file = out_file
                conv.run()
                cmd = 'mv %s %s' % (in_file, op.join(op.abspath("Original"), name + ext))
                os.system(cmd)
                print(cmd)
            out_files.append(out_file)
        elif len(in_file) > 1:
            print("Multiple files found using %s" % search_str)
        else:
            print("Couldn't find anything using %s" % search_str)

    print("Successfully conformed %d files" % len(out_files))
    print(out_files)
    return out_files


if __name__ == '__main__':
    # Should probably use argparse for this
    base_name = sys.argv[1]
    conformAll(base_name)
