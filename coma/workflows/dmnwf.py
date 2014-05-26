import os
import os.path as op
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.mrtrix as mrtrix
import nipype.interfaces.cmtk as cmtk
from nipype.workflows.misc.utils import select_aparc

fsl.FSLCommand.set_default_output_type('NIFTI')

from coma.interfaces import nonlinfit_fn
from coma.dti.basic import damaged_brain_dti_processing
from coma.workflow.dmn import create_paired_tract_analysis_wf
from coma.workflow.pet import create_pet_quantification_wf
from coma.labels import dmn_labels_combined

def create_dmn_pipeline_step1(name="dmn_step1", auto_reorient=True):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subjects_dir",
                                                 "subject_id",
                                                 "dwi",
                                                 "bvecs",
                                                 "bvals",
                                                 "fdgpet"]),
        name="inputnode")

    output_fields = ["fa", "rgb_fa", "md", "csdeconv", "tracts_tck", "rois", "t1",
        "t1_brain", "wmmask_dtispace", "fa_t1space", "summary", "filtered_tractographies",
        "fiber_label_file", "intersection_matrix_mat_file"]

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name="outputnode")

    dmn_labels_if = util.Function(input_names=["in_file", "out_filename"],
        output_names=["out_file"], function=dmn_labels_combined)
    dmn_labelling = pe.Node(interface=dmn_labels_if, name='dmn_labelling')

    dtiproc = damaged_brain_dti_processing("dtiproc")
    petquant = create_pet_quantification_wf("petquant")

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect([(inputnode, dtiproc, [("subjects_dir", "inputnode.subjects_dir"),
                                                ("subject_id", "inputnode.subject_id"),
                                                ("dwi", "inputnode.dwi"),
                                                ("bvecs", "inputnode.bvecs"),
                                                ("bvals", "inputnode.bvals")])
                                              ])

    workflow.connect([(dtiproc, petquant, [("outputnode.t1", "inputnode.t1"),
                                           ("outputnode.wm", "inputnode.wm"),
                                           ("outputnode.gm", "inputnode.gm"),
                                           ("outputnode.csf", "inputnode.csf")])
                                              ])

    workflow.connect([(inputnode, petquant,[("fdgpet","inputnode.pet")])])
    workflow.connect([(dtiproc, dmn_labelling,[("outputnode.rois","in_file")])])
    workflow.connect([(dmn_labelling, petquant,[("out_file","inputnode.rois")])])
    return workflow


def create_dmn_pipeline_step2(name="dmn_step1", auto_reorient=True):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["track_file",
                                                 "fa",
                                                 "md",
                                                 "roi_file"]),
        name="inputnode")

    # fiber tracking
    connectivity = create_paired_tract_analysis_wf("connectivity")

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect([(inputnode, connectivity, [("track_file", "inputnode.track_file"),
                                                ("fa", "inputnode.fa"),
                                                ("md", "inputnode.md"),
                                                ("roi_file", "inputnode.roi_file")])
                                              ])
    return workflow

