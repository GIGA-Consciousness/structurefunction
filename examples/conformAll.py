#!/usr/bin/env python
import os, sys
import string

def conformAll(base_name):
    import os.path as op
    import nipype.interfaces.freesurfer as fs
    from nipype.utils.filemanip import split_filename
    import glob
    list_of_exts = ["_T1"    , "_T1brain", "_PreCoTh_rois",
                    "_fdgpet", "_wmmask" , "_cortex"]
    out_files = []

    print("Base name: '%s'" % base_name)

    for filetype in list_of_exts:
        print("Searching for '%s'" % filetype)
        search_str = op.abspath(base_name + "*" + filetype + ".*")
        in_file = glob.glob(search_str)
        if in_file and len(in_file) == 1:
            in_file = in_file[0]
            print("Found %s" % in_file)
            _, name, ext = split_filename(in_file)
            out_file = op.abspath("conform_" + name + ext)
            conv = fs.MRIConvert()
            conv.inputs.conform = True
            conv.inputs.no_change = True
            conv.inputs.in_file = in_file
            conv.inputs.out_file = out_file
            conv.run()
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
