import os
import os.path as op
import glob
import subprocess
from write_ctab import write_ctab

subject = "fsaverage"
hemi = "lh"
out_ctab = "myctab.ctab"
path_to_labels = op.abspath("/media/Mobsol/Dropbox/HostedData/FSAverageLausanne2008")
out_annot_file = "Lausanne1015_fsavg.annot"

#label_files = glob.glob(op.join(path_to_labels, "regenerated_%s_500/rh.*" % hemi))
label_files = glob.glob(op.join(path_to_labels, "%s_500_clean/%s.*" % (hemi, hemi)))
#label_files = glob.glob(op.join(os.environ["SUBJECTS_DIR"], "%s/label/LausanneLabels/%s.*" % (subject, hemi)))
label_str = " --l ".join(label_files)

write_ctab(label_files, out_ctab)

#to_delete = op.join(os.environ["SUBJECTS_DIR"], "%s/label/%s." % (subject, hemi))
#to_delete = to_delete + out_annot_file + ".annot"
#subprocess.call(["rm", to_delete])

call_list = ["mris_label2annot", "--s", subject, "--l", label_str,
    " --a", out_annot_file, "--hemi", hemi, "--ctab", out_ctab]
print(" ".join(call_list))
os.system(" ".join(call_list))
