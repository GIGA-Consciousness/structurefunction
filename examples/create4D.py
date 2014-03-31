#!/usr/bin/env python
import sys

def create4Ddwi(name,out):
	import glob
	print("Looking for %s" % name)
	name = str(name)
	out = str(out)
	a = glob.glob(name + "*.img")
	assert(len(a) >= 30)
	print("%i volumes found" % len(a))
	a.sort(key=lambda x: x[-8:])
	import nibabel as nb
	print("Concatenating %i files" % len(a))
	b = nb.concat_images(a)
	b.get_header()
	out_file = out+".nii.gz"
	print("Writing to %s" % out_file)
	nb.save(b, out_file)
	return out_file

if __name__ == '__main__':
	name, out = sys.argv[1:]
	create4Ddwi(name, out)