import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl

fsl.FSLCommand.set_default_output_type('NIFTI')

from coma.workflows.dti.basic import damaged_brain_dti_processing
from coma.workflows.dmn import create_paired_tract_analysis_wf
from coma.workflows.pet import create_pet_quantification_wf
from coma.labels import dmn_labels_combined
from coma.helpers import add_subj_name_to_rois


def create_reg_and_label_wf(name="reg_wf"):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subject_id",
                                                 "aparc_aseg",
                                                 "fa",
                                                 "wm_mask"]),
        name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["dwi_to_t1_matrix",
                                                 "t1_to_dwi_matrix",
                                                 "rois_to_dwi",
                                                 "rois"]),
        name="outputnode")

    coregister = pe.Node(interface=fsl.FLIRT(dof=12), name = 'coregister')
    coregister.inputs.cost = ('normmi')
    coregister.inputs.interp = 'trilinear'

    convertxfm = pe.Node(interface=fsl.ConvertXFM(), name='convertxfm')
    convertxfm.inputs.invert_xfm = True

    rois_to_dwispace = pe.Node(interface=fsl.ApplyXfm(), name='rois_to_dwispace')
    rois_to_dwispace.inputs.apply_xfm = True
    rois_to_dwispace.inputs.interp = 'nearestneighbour'

    coreg_noresamp = pe.Node(interface=fsl.FLIRT(dof=12), name = 'coreg_noresamp')
    coreg_noresamp.inputs.cost = ('normmi')
    coreg_noresamp.inputs.no_resample = True

    invertxfm = pe.Node(interface=fsl.ConvertXFM(), name = 'invertxfm')
    invertxfm.inputs.invert_xfm = True

    dmn_labels_if = util.Function(input_names=["in_file", "out_filename"],
                                  output_names=["out_file"], function=dmn_labels_combined)
    dmn_labelling = pe.Node(interface=dmn_labels_if, name='dmn_labelling')

    workflow = pe.Workflow(name=name)
    workflow.connect(
        [(inputnode, dmn_labelling, [(('subject_id', add_subj_name_to_rois), 'out_filename')])])
    workflow.connect([(inputnode, dmn_labelling, [("aparc_aseg", "in_file")])])

    workflow.connect([(inputnode, coregister,[("fa","in_file")])])
    workflow.connect([(inputnode, coregister,[('wm_mask','reference')])])
    workflow.connect([(coregister, invertxfm,[("out_matrix_file","in_file")])])

    workflow.connect([(inputnode, coreg_noresamp,[("wm_mask","in_file")])])
    workflow.connect([(inputnode, coreg_noresamp,[('fa','reference')])])

    workflow.connect([(dmn_labelling, rois_to_dwispace, [("out_file", "in_file")])])
    workflow.connect([(inputnode, rois_to_dwispace, [("fa", "reference")])])
    workflow.connect([(coreg_noresamp, rois_to_dwispace,
        [('out_matrix_file','in_matrix_file')])])

    workflow.connect(
        [(dmn_labelling, outputnode, [("out_file", "rois")])])
    workflow.connect(
        [(coregister, outputnode, [("out_matrix_file", "t1_to_dwi_matrix")])])
    workflow.connect(
        [(invertxfm, outputnode, [("out_file", "dwi_to_t1_matrix")])])
    workflow.connect(
        [(rois_to_dwispace, outputnode, [("out_file", "rois_to_dwi")])])
    return workflow


def create_dmn_pipeline_step1(name="dmn_step1", auto_reorient=True):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["subjects_dir",
                                                 "subject_id",
                                                 "dwi",
                                                 "bvecs",
                                                 "bvals",
                                                 "fdgpet"]),
        name="inputnode")

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=["fa",
                                                 "rgb_fa",
                                                 "md",
                                                 "csdeconv",
                                                 "tracts_tck", "rois", "t1",
                                                 "t1_brain", "wmmask_dtispace", "fa_t1space", "summary", "filtered_tractographies",
                                                 "fiber_label_file", "intersection_matrix_mat_file"]),
        name="outputnode")

    dtiproc = damaged_brain_dti_processing("dtiproc")
    reg_label = create_reg_and_label_wf("reg_label")
    petquant = create_pet_quantification_wf("petquant", segment_t1=False)

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect(
        [(inputnode, dtiproc, [("subjects_dir", "inputnode.subjects_dir"),
                               ("subject_id", "inputnode.subject_id"),
                               ("dwi", "inputnode.dwi"),
                               ("bvecs", "inputnode.bvecs"),
                               ("bvals", "inputnode.bvals")])
         ])

    workflow.connect([(inputnode, reg_label, [("subject_id", "inputnode.subject_id")])])
    
    workflow.connect([(dtiproc, reg_label, [("outputnode.wm_prob", "inputnode.wm_mask"),
                                            ("outputnode.fa", "inputnode.fa"),
                                            ("outputnode.aparc_aseg", "inputnode.aparc_aseg"),
                      ])])



    workflow.connect([(inputnode, petquant, [("fdgpet", "inputnode.pet")])])
    workflow.connect([(reg_label, petquant, [("outputnode.rois", "inputnode.rois")])])
    workflow.connect([(dtiproc, petquant, [("outputnode.t1", "inputnode.t1"),
                                           ("outputnode.wm_prob", "inputnode.wm_prob"),
                                           ("outputnode.gm_prob", "inputnode.gm_prob"),
                                           ("outputnode.csf_prob", "inputnode.csf_prob"),
                                           ])])

    '''
    Connect outputnode
    '''
    #workflow.connect([(petquant, petquant, [("rois", "inputnode.rois")])])
    # "out_files",
    #                                                  "pet_to_t1",
    #                                                  "corrected_pet_to_t1",
    #                                                  "pet_results_npz",
    #                                                  "pet_results_mat"
    # "single_fiber_mask",
    #                                                  "fa",
    #                                                  "rgb_fa",
    #                                                  "md",
    #                                                  "mode",
    #                                                  "t1",
    #                                                  "t1_brain",
    #                                                  "wm_mask",
    #                                                  "term_mask",
    #                                                  "aparc_aseg",
    #                                                  "tissue_class_files",
    #                                                  "gm_prob",
    #                                                  "wm_prob",
    #                                                  "csf_prob"
    # "dwi_to_t1_matrix",
    #                                                  "t1_to_dwi_matrix",
    #                                                  "rois_to_dwi",
    #                                                  "rois"
    return workflow


def create_dmn_pipeline_step2(name="dmn_step2", auto_reorient=True):
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=["track_file",
                                                 "fa",
                                                 "md",
                                                 "roi_file"]),
        name="inputnode")

    # fiber tracking
    connectivity = create_paired_tract_analysis_wf("connectivity")

    workflow = pe.Workflow(name=name)
    workflow.base_dir = name

    workflow.connect(
        [(inputnode, connectivity, [("track_file", "inputnode.track_file"),
                                    ("fa", "inputnode.fa"),
                                    ("md", "inputnode.md"),
                                    ("roi_file", "inputnode.roi_file")])
         ])
    return workflow
